r"""Cosmos 3 attention on ColossalAI's two-way flat-attention kernel.

Cosmos 3 keeps its attention core in a replaceable ``processor`` object, and its forward only calls
``self.processor(self, und_seq, gen_seq, rotary_emb)``. :class:`ColossalTwoWayAttnProcessor` keeps the stock
``Cosmos3AttnProcessor`` prologue -- per-pathway projections, q/k norms, 3D RoPE -- and hands the understanding (AR)
prefix and the generation (DM) suffix to
:func:`~colossalai.shardformer.layer.two_way_attn.two_way_flat_attention` in one call instead of two
``dispatch_attention_fn`` calls.

The per-sample boundaries of a packed batch travel beside the forward arguments as a
:class:`~colossalai.shardformer.layer.two_way_attn.TwoWayPackedMeta`; the layout is described on the processor class.
Only the attribute contract of the stock processor is used, so this module imports without diffusers.
"""

from types import MethodType
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from colossalai.logging import get_dist_logger

from ..layer._operation import gather_forward_reducescatter_backward, packed_all_to_all_single
from ..layer.two_way_attn import TwoWayPackedMeta, _load_cuda_varlen_attention, two_way_flat_attention

__all__ = [
    "ColossalTwoWayAttnProcessor",
    "ColossalTwoWayCPAttnProcessor",
    "Cosmos3ContextParallel",
    "Cosmos3TwoWayAttnProcessor",
    "enable_cosmos3_context_parallel",
    "enable_cosmos3_selective_checkpointing",
    "set_cosmos3_packed_meta",
    "clear_cosmos3_packed_meta",
]

_ATTENTION_CLS = "Cosmos3PackedMoTAttention"
_DECODER_LAYER_CLS = "Cosmos3VLTextMoTDecoderLayer"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half, matching the diffusers Cosmos 3 processor."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class ColossalTwoWayAttnProcessor:
    """Drop-in replacement for diffusers' ``Cosmos3AttnProcessor`` backed by the two-way kernel.

    A pack of ``N`` samples is laid out as a global AR block followed by a global DM block, sample ``i`` owning
    ``ar_lens[i]`` contiguous AR tokens and ``dm_lens[i]`` contiguous DM tokens. That is the split ``und_len`` already
    describes, so the per-sample boundaries are set on the processor -- one instance is shared by every decoder layer
    -- before the forward; without them the whole pack counts as one sample, the stock batch-size-one behaviour.

    Args:
        backend (str, optional): ``"naive"`` for the PyTorch reference, ``"npu"`` for the Ascend fusion path or
            ``"cuda"`` for PyTorch's native packed CUDA kernel. Defaults to ``"naive"`` so the processor also runs on
            CPU.
        packed_meta (Optional[TwoWayPackedMeta], optional): Per-sample AR/DM lengths of the pack. When omitted, every
            forward is treated as a single sample whose lengths are read from the two input tensors.
        scale (Optional[float], optional): Softmax scale forwarded to the two-way kernel. Defaults to None.
        dropout_p (float, optional): Attention dropout. The CUDA backend rejects a non-zero value because the native
            packed kernel has no dropout. Defaults to 0.0.

    Raises:
        ValueError: If ``backend`` is unknown.
    """

    # Kept for drop-in parity with the stock processor: diffusers' `set_attention_backend` writes here. The two
    # attention calls are chosen by `backend` instead, so a value written here is recorded and otherwise unused.
    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        backend: str = "naive",
        packed_meta: Optional[TwoWayPackedMeta] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> None:
        if backend not in ("naive", "npu", "cuda"):
            raise ValueError(f"backend must be 'naive', 'npu' or 'cuda', got {backend!r}.")
        self.backend = backend
        self.scale = scale
        self.dropout_p = dropout_p
        # Warm the packed CUDA kernel here, not in the first forward: the loader imports through importlib, which
        # torch.compile refuses to trace, and the processor is built before the model is compiled.
        if backend == "cuda":
            _load_cuda_varlen_attention()
        self.packed_meta: Optional[TwoWayPackedMeta] = None
        if packed_meta is not None:
            self.set_packed_meta(packed_meta)

    def set_packed_meta(self, packed_meta) -> None:
        """Set the current pack's per-sample boundaries.

        Args:
            packed_meta (Optional[TwoWayPackedMeta]): The metadata of the next pack, or a
                :class:`~colossalai.shardformer.modeling.cosmos_mot.CosmosTwoWayLayout` carrying one. ``None`` restores
                the single-sample fallback.

        Raises:
            TypeError: If the value is neither ``None``, a ``TwoWayPackedMeta`` nor an object with a ``meta`` field.
        """
        if packed_meta is not None and not isinstance(packed_meta, TwoWayPackedMeta):
            # Accept the packer-facing layout object as well, so callers do not have to reach into `.meta`.
            packed_meta = getattr(packed_meta, "meta", packed_meta)
        if packed_meta is not None and not isinstance(packed_meta, TwoWayPackedMeta):
            raise TypeError(
                "packed_meta must be a TwoWayPackedMeta, a CosmosTwoWayLayout or None, got "
                f"{type(packed_meta).__name__}."
            )
        self.packed_meta = packed_meta

    def clear_packed_meta(self) -> None:
        """Drop the pack metadata, restoring the single-sample fallback for the next forward."""
        self.set_packed_meta(None)

    def _resolve_meta(self, num_ar_tokens: int, num_dm_tokens: int) -> TwoWayPackedMeta:
        if self.packed_meta is None:
            return TwoWayPackedMeta([num_ar_tokens], [num_dm_tokens])
        meta = self.packed_meta
        if meta.total_ar_tokens != num_ar_tokens or meta.total_dm_tokens != num_dm_tokens:
            raise ValueError(
                f"The packed metadata describes {meta.total_ar_tokens} AR and {meta.total_dm_tokens} DM tokens, but "
                f"the attention inputs carry {num_ar_tokens} AR and {num_dm_tokens} DM tokens. Set the metadata from "
                f"the same pack that produced the inputs."
            )
        return meta

    def _project(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        """Per-pathway q/k/v projections, q/k norms and 3D RoPE, exactly as the stock ``Cosmos3AttnProcessor``.

        Shared by the single-device and the context-parallel processors, which differ only in what they do with the
        returned tensors: a plain run hands them to the two-way kernel, a Ulysses run exchanges sequence for heads
        first. Everything here is token-local, so it gives the same answer on a sequence shard.

        Returns:
            tuple: ``(q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen)``, each of shape
            ``[tokens, heads, head_dim]``; ``k_und_for_gen`` is None unless the model normalizes the AR keys a second
            time for the generation pathway.
        """
        q_und = attn.to_q(und_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_und = attn.to_k(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_und = attn.to_v(und_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        q_gen = attn.add_q_proj(gen_seq).view(-1, attn.num_attention_heads, attn.head_dim)
        k_gen = attn.add_k_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)
        v_gen = attn.add_v_proj(gen_seq).view(-1, attn.num_key_value_heads, attn.head_dim)

        q_und = attn.norm_q(q_und)
        k_und = attn.norm_k(k_und)
        # Cosmos Edge normalizes the AR keys once more for the generation pathway; other variants leave this None, in
        # which case the generation pass reuses the AR key block as-is.
        k_norm_und_for_gen = getattr(attn, "k_norm_und_for_gen", None)
        k_und_for_gen = None if k_norm_und_for_gen is None else k_norm_und_for_gen(k_und)
        q_gen = attn.norm_added_q(q_gen)
        k_gen = attn.norm_added_k(k_gen)

        # Apply rotary position embeddings per pathway.
        cos_und, sin_und, cos_gen, sin_gen = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + _rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + _rotate_half(k_und) * sin_und
        if k_und_for_gen is not None:
            k_und_for_gen = k_und_for_gen * cos_und + _rotate_half(k_und_for_gen) * sin_und
        cos_gen = cos_gen.unsqueeze(1)
        sin_gen = sin_gen.unsqueeze(1)
        q_gen = q_gen * cos_gen + _rotate_half(q_gen) * sin_gen
        k_gen = k_gen * cos_gen + _rotate_half(k_gen) * sin_gen
        return q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen

    def __call__(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen = self._project(
            attn, und_seq, gen_seq, rotary_emb
        )

        meta = self._resolve_meta(q_und.shape[0], q_gen.shape[0])
        und_out, gen_out = two_way_flat_attention(
            q_und,
            k_und,
            v_und,
            q_gen,
            k_gen,
            v_gen,
            meta,
            backend=self.backend,
            k_ar_for_gen=k_und_for_gen,
            scale=self.scale,
            dropout_p=self.dropout_p,
        )

        # Per-pathway output projection.
        return attn.to_out(und_out.flatten(-2, -1)), attn.to_add_out(gen_out.flatten(-2, -1))


# The model-prefixed spelling reads better from framework code; the historical name is kept for the SFT scripts.
Cosmos3TwoWayAttnProcessor = ColossalTwoWayAttnProcessor


class ColossalTwoWayCPAttnProcessor(ColossalTwoWayAttnProcessor):
    """Two-way attention under Ulysses context parallelism.

    Each rank starts with a contiguous slice of the packed sequence, so the projections, the q/k norms and the RoPE
    are already correct on it. Attention then needs the opposite layout: every query of the full sequence, for a
    fraction of the heads. Three all-to-alls per pathway (plus the optional AR-key-for-generation tensor) swap the two
    -- sequence gathered, heads scattered -- the two-way kernel runs on the full pack with the *global* per-sample
    metadata, and two all-to-alls restore the sequence-sharded layout. That is the Ulysses scheme the report uses
    (Sec. 5.2.3), and it keeps the two-way kernel unchanged: after the exchange each rank holds whole per-sample AR
    and DM blocks for its heads.

    Two details make it exact rather than approximate. The caller pads each pathway to a multiple of the context
    parallel size before sharding; the exchange therefore hands back padded blocks, which are trimmed to the real
    per-sample totals before the kernel and re-padded after it, so no padding token ever enters an attention score.
    And the per-sample boundaries are the ones the packer installed on the processor for the whole pack, not a
    re-derivation from the shard.

    Args:
        backend (str, optional): ``"naive"``, ``"npu"`` or ``"cuda"``, forwarded to the two-way kernel. Defaults to
            ``"naive"``.
        process_group (ProcessGroup, optional): The context parallel group. ``None`` is only valid on a single-rank
            world; the exchange is a no-op with a context parallel size of one. Defaults to None.
        packed_meta (Optional[TwoWayPackedMeta], optional): See :class:`ColossalTwoWayAttnProcessor`.
        scale (Optional[float], optional): Softmax scale forwarded to the two-way kernel. Defaults to None.
        dropout_p (float, optional): Attention dropout. Defaults to 0.0.
    """

    def __init__(
        self,
        backend: str = "naive",
        process_group=None,
        packed_meta: Optional[TwoWayPackedMeta] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> None:
        if process_group is None and dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError(
                "Cosmos 3 context parallelism needs an explicit process group when the world size is greater than "
                "1; pass the plugin's `sequence_parallel_process_group`."
            )
        super().__init__(backend=backend, packed_meta=packed_meta, scale=scale, dropout_p=dropout_p)
        self.process_group = process_group
        self.cp_size = 1 if process_group is None else dist.get_world_size(process_group)
        # Real (unpadded) lengths of the pack, stamped by the shard seam of the forward that is running.
        self.und_real: Optional[int] = None
        self.gen_real: Optional[int] = None

    def set_sequence_lengths(self, und_real: int, gen_real: int) -> None:
        """Record the real per-pathway totals of the pack whose shards the attention layers are about to see."""
        self.und_real, self.gen_real = int(und_real), int(gen_real)

    def _exchange(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the sequence and scatter the heads: ``[S_local, H, D]`` to ``[S_local * cp, H / cp, D]``."""
        if self.cp_size == 1:
            return x
        tokens, heads, head_dim = x.shape
        if tokens == 0:
            return x.new_empty((0, heads // self.cp_size, head_dim))
        if heads % self.cp_size != 0:
            raise ValueError(
                f"Cosmos3 attention heads ({heads}) must be divisible by the context parallel size ({self.cp_size})."
            )
        inner = heads // self.cp_size
        # Chunk j of the send buffer is destined for rank j, which keeps head chunk j.
        send = x.reshape(tokens, self.cp_size, inner, head_dim).permute(1, 0, 2, 3).contiguous()
        recv = packed_all_to_all_single(send.view(self.cp_size, -1), self.process_group)
        # Chunk i of the receive buffer comes from rank i, whose tokens precede this rank's in the pack, so the
        # sequence is already in pack order once the chunks are concatenated.
        return recv.view(self.cp_size, tokens, inner, head_dim).reshape(tokens * self.cp_size, inner, head_dim)

    def _exchange_back(self, x: torch.Tensor) -> torch.Tensor:
        """Gather the heads and scatter the sequence: ``[S_padded, H / cp, D]`` to ``[S_padded / cp, H, D]``."""
        if self.cp_size == 1:
            return x
        tokens, inner, head_dim = x.shape
        if tokens == 0:
            return x.new_empty((0, inner * self.cp_size, head_dim))
        local = tokens // self.cp_size
        # Sequence chunk j is destined for rank j, which collects every rank's head chunk for it.
        send = x.reshape(self.cp_size, local, inner, head_dim).contiguous()
        recv = packed_all_to_all_single(send.view(self.cp_size, -1), self.process_group)
        return recv.view(self.cp_size, local, inner, head_dim).permute(1, 0, 2, 3).reshape(
            local, inner * self.cp_size, head_dim
        )

    def __call__(
        self,
        attn,
        und_seq: torch.Tensor,
        gen_seq: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = self.packed_meta
        if meta is None:
            raise RuntimeError(
                "Context-parallel Cosmos 3 attention needs the per-sample boundaries of the pack: call "
                "`policy.set_packed_meta(...)` from the packer before the forward."
            )
        if self.und_real is None or self.gen_real is None:
            raise RuntimeError(
                "Context-parallel Cosmos 3 attention needs the real sequence lengths recorded by the CP shard seam; "
                "run the model through `Cosmos3ContextParallel.shard` before the decoder layers."
            )
        if meta.total_ar_tokens != self.und_real or meta.total_dm_tokens != self.gen_real:
            raise ValueError(
                f"The packed metadata describes {meta.total_ar_tokens} AR and {meta.total_dm_tokens} DM tokens, but "
                f"the CP shard seam carries {self.und_real} AR and {self.gen_real} DM tokens. Set the metadata from "
                f"the same pack that produced the inputs."
            )
        expected_ar_tokens = (self.und_real + self.cp_size - 1) // self.cp_size
        expected_dm_tokens = (self.gen_real + self.cp_size - 1) // self.cp_size
        if und_seq.shape[0] != expected_ar_tokens or gen_seq.shape[0] != expected_dm_tokens:
            raise ValueError(
                f"The CP shards should carry {expected_ar_tokens} AR and {expected_dm_tokens} DM tokens for a context "
                f"parallel size of {self.cp_size}, but the attention inputs carry {und_seq.shape[0]} and "
                f"{gen_seq.shape[0]}."
            )
        q_und, k_und, v_und, q_gen, k_gen, v_gen, k_und_for_gen = self._project(
            attn, und_seq, gen_seq, rotary_emb
        )

        # Trim the padding the shard seam appended, so the kernel sees exactly the tokens the packer described.
        ar, dm = meta.total_ar_tokens, meta.total_dm_tokens
        q_und = self._exchange(q_und)[:ar]
        k_und = self._exchange(k_und)[:ar]
        v_und = self._exchange(v_und)[:ar]
        if k_und_for_gen is not None:
            k_und_for_gen = self._exchange(k_und_for_gen)[:ar]
        q_gen = self._exchange(q_gen)[:dm]
        k_gen = self._exchange(k_gen)[:dm]
        v_gen = self._exchange(v_gen)[:dm]

        und_out, gen_out = two_way_flat_attention(
            q_und,
            k_und,
            v_und,
            q_gen,
            k_gen,
            v_gen,
            meta,
            backend=self.backend,
            k_ar_for_gen=k_und_for_gen,
            scale=self.scale,
            dropout_p=self.dropout_p,
        )

        # Re-pad, so the second exchange moves the same number of tokens on every rank, then hand back this rank's
        # sequence slice.
        und_out = self._exchange_back(self._pad_output(und_out, und_seq.shape[0]))
        gen_out = self._exchange_back(self._pad_output(gen_out, gen_seq.shape[0]))
        return attn.to_out(und_out.flatten(-2, -1)), attn.to_add_out(gen_out.flatten(-2, -1))

    def _pad_output(self, x: torch.Tensor, local_tokens: int) -> torch.Tensor:
        padded = local_tokens * self.cp_size
        if x.shape[0] == padded:
            return x
        return torch.cat([x, x.new_zeros((padded - x.shape[0],) + tuple(x.shape[1:]))], dim=0)


class Cosmos3ContextParallel:
    """The two seams that put a ``Cosmos3OmniTransformer`` into Ulysses context parallelism.

    The model calls ``_cp_shard_fn`` once per forward, right after the embeddings and before the decoder stack, and
    ``_cp_gather_fn`` once before the output projections. The first pads each pathway to a multiple of the context
    parallel size and keeps this rank's contiguous slice; the second returns the full per-pathway output, trimmed to
    the real length. Everything between them -- norms, MLP, residuals -- is token-local and runs on the slice, while
    attention crosses the group through :class:`ColossalTwoWayCPAttnProcessor`.

    The pack metadata stays global: every rank runs the embedding stage on the whole pack and every rank decodes the
    whole pack, so the loss, the indexes and the per-sample bookkeeping are identical to a single-device run. Every
    rank in the group must build the same pack -- same samples, same noise and same timesteps -- because the shards
    are concatenated into one sequence; the seam checks the real lengths and errors on a mismatch.

    Args:
        process_group (ProcessGroup, optional): The context parallel group. ``None`` is only valid on a single-rank
            world; a multi-rank world has to pass the group explicitly so a missing group cannot silently turn
            context parallelism into a no-op. Defaults to None.
        backend (str, optional): Two-way kernel backend. Defaults to ``"naive"``.
        packed_meta (Optional[TwoWayPackedMeta], optional): Pack metadata to install on the CP processor. Defaults to
            None, i.e. the caller sets it from the packer before the first forward.
        scale (Optional[float], optional): Softmax scale forwarded to the CP processor. Defaults to None.
        dropout_p (float, optional): Attention dropout forwarded to the CP processor. Defaults to 0.0.
    """

    def __init__(
        self,
        process_group=None,
        backend: str = "naive",
        packed_meta: Optional[TwoWayPackedMeta] = None,
        scale: Optional[float] = None,
        dropout_p: float = 0.0,
    ) -> None:
        if process_group is None and dist.is_initialized() and dist.get_world_size() > 1:
            raise ValueError(
                "Cosmos 3 context parallelism needs an explicit process group when the world size is greater than "
                "1; pass the plugin's `sequence_parallel_process_group`."
            )
        self.process_group = process_group
        self.cp_size = 1 if process_group is None else dist.get_world_size(process_group)
        self.cp_rank = 0 if process_group is None else dist.get_rank(process_group)
        self.processor = ColossalTwoWayCPAttnProcessor(
            backend=backend,
            process_group=process_group,
            packed_meta=packed_meta,
            scale=scale,
            dropout_p=dropout_p,
        )
        self._und_real: Optional[int] = None
        self._gen_real: Optional[int] = None

    @property
    def packed_meta(self) -> Optional[TwoWayPackedMeta]:
        """The pack metadata currently installed on the CP processor, or ``None``."""
        return self.processor.packed_meta

    def set_packed_meta(self, packed_meta) -> None:
        """Delegate the next pack's per-sample boundaries to the CP processor."""
        self.processor.set_packed_meta(packed_meta)

    def clear_packed_meta(self) -> None:
        """Delegate clearing the pack metadata to the CP processor."""
        self.processor.clear_packed_meta()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        remainder = x.shape[0] % self.cp_size
        if remainder == 0:
            return x
        return torch.cat([x, x.new_zeros((self.cp_size - remainder,) + tuple(x.shape[1:]))], dim=0)

    def _shard(self, x: torch.Tensor) -> torch.Tensor:
        return torch.chunk(self._pad(x), self.cp_size, dim=0)[self.cp_rank]

    def _gather(self, x: torch.Tensor) -> torch.Tensor:
        if self.cp_size == 1:
            return x
        return gather_forward_reducescatter_backward(x, self.process_group, dim=0)

    def shard(self, und_seq, gen_seq, rotary_emb):
        """The ``_cp_shard_fn`` seam: pad each pathway, keep this rank's contiguous token slice."""
        self._und_real, self._gen_real = int(und_seq.shape[0]), int(gen_seq.shape[0])
        if self.cp_size > 1:
            # Context parallelism splits one pack, so every rank has to see the same real lengths. Checking them
            # here costs one tiny all-gather per forward and turns a silently mismatched pack into an error before
            # any attention collective can hang.
            lengths = torch.tensor(
                [self._und_real, self._gen_real], device=und_seq.device, dtype=torch.long
            )
            gathered = [torch.empty_like(lengths) for _ in range(self.cp_size)]
            dist.all_gather(gathered, lengths, group=self.process_group)
            if any(tuple(item.tolist()) != (self._und_real, self._gen_real) for item in gathered):
                raise ValueError(
                    "Every rank in the context parallel group must see the same packed sequence; got "
                    f"{(self._und_real, self._gen_real)} on this rank and "
                    f"{[tuple(item.tolist()) for item in gathered]} across the group."
                )
        self.processor.set_sequence_lengths(self._und_real, self._gen_real)
        return (
            self._shard(und_seq),
            self._shard(gen_seq),
            tuple(self._shard(tensor) for tensor in rotary_emb),
        )

    def gather(self, und_out, gen_out):
        """The ``_cp_gather_fn`` seam: reassemble each pathway and drop the padding."""
        return (
            self._gather(und_out)[: self._und_real],
            self._gather(gen_out)[: self._gen_real],
        )


def enable_cosmos3_context_parallel(
    model: torch.nn.Module,
    process_group=None,
    backend: str = "naive",
    packed_meta: Optional[TwoWayPackedMeta] = None,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
) -> Cosmos3ContextParallel:
    """Install Ulysses context parallelism on a Cosmos 3 omni transformer.

    Sets the context-parallel attention processor on every attention module, and wires the model's two CP seams to a
    :class:`Cosmos3ContextParallel` instance, which the caller keeps in order to call ``set_packed_meta`` on it.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer.
        process_group (ProcessGroup, optional): The context parallel group. ``None`` is only valid on a single-rank
            world; a multi-rank world has to pass the group explicitly. Defaults to None.
        backend (str, optional): Two-way kernel backend. Defaults to ``"naive"``.
        packed_meta (Optional[TwoWayPackedMeta], optional): Pack metadata to install on the CP processor. Defaults to
            None, i.e. the caller sets it from the packer before the first forward.
        scale (Optional[float], optional): Softmax scale forwarded to the CP processor. Defaults to None.
        dropout_p (float, optional): Attention dropout forwarded to the CP processor. Defaults to 0.0.

    Returns:
        Cosmos3ContextParallel: The helper owning the processor and the seams.

    Raises:
        RuntimeError: If the model has no CP seams, i.e. the installed diffusers predates them; silently setting the
            attributes would leave every rank with the full sequence and no error.
        ValueError: If ``process_group`` is None on a world with more than one rank.
    """
    if not hasattr(model, "_cp_shard_fn") or not hasattr(model, "_cp_gather_fn"):
        raise RuntimeError(
            f"{type(model).__name__} has no Cosmos 3 context-parallel seams (`_cp_shard_fn`/`_cp_gather_fn`); "
            "context parallelism needs a diffusers build that exposes them."
        )
    context_parallel = Cosmos3ContextParallel(
        process_group=process_group,
        backend=backend,
        packed_meta=packed_meta,
        scale=scale,
        dropout_p=dropout_p,
    )
    installed = 0
    for module in model.modules():
        if module.__class__.__name__ != _ATTENTION_CLS:
            continue
        module.set_processor(context_parallel.processor)
        installed += 1
    if installed == 0:
        raise RuntimeError(
            f"No {_ATTENTION_CLS} module found in {type(model).__name__}; context parallelism only applies to a "
            "Cosmos 3 omni transformer."
        )
    model._cp_shard_fn = context_parallel.shard
    model._cp_gather_fn = context_parallel.gather
    return context_parallel


def set_cosmos3_packed_meta(model: torch.nn.Module, packed_meta, strict: bool = False) -> int:
    """Set the pack metadata on every Cosmos 3 attention processor in ``model``.

    This is the plugin-agnostic path: it walks the module tree, so it also works through a ``Booster`` wrapper
    (``ModelWrapper`` registers the wrapped module as a child) and when the policy object is not at hand.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        packed_meta (Optional[TwoWayPackedMeta]): Metadata for the next forward; see
            :meth:`ColossalTwoWayAttnProcessor.set_packed_meta`.
        strict (bool, optional): Raise if no Cosmos 3 attention processor was found. A training entry point should
            pass ``strict=True`` so a model that was not sharded through the policy fails loudly instead of training
            with the stock single-sample processor. Defaults to False.

    Returns:
        int: Number of attention processors updated.

    Raises:
        RuntimeError: If ``strict`` is true and no processor was updated.
    """
    updated = 0
    for module in model.modules():
        if module.__class__.__name__ != _ATTENTION_CLS:
            continue
        processor = getattr(module, "processor", None)
        setter = getattr(processor, "set_packed_meta", None)
        if setter is not None:
            setter(packed_meta)
            updated += 1
    if strict and updated == 0:
        raise RuntimeError(
            f"No Cosmos 3 attention processor was found in {type(model).__name__}. Boost the model through "
            f"Cosmos3OmniPolicy before setting pack metadata, or call this helper on the unwrapped transformer."
        )
    return updated


def clear_cosmos3_packed_meta(model: torch.nn.Module, strict: bool = False) -> int:
    """Remove pack metadata from every Cosmos 3 attention processor in ``model``.

    Call this after a packed step has completed. It restores the stock single-sample behaviour, so a later forward
    cannot silently reuse the previous pack's boundaries.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        strict (bool, optional): Raise if no Cosmos 3 attention processor was found. Defaults to False.

    Returns:
        int: Number of attention processors updated.
    """
    return set_cosmos3_packed_meta(model, None, strict=strict)


def _checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb):
    """One Cosmos 3 decoder layer with its two MLPs under activation checkpointing.

    Same statements as the stock forward, in the same order: the input norms, the attention call, the two residuals.
    Only the checkpoint boundary moves -- around the MLP pair instead of around the whole layer -- so the backward
    pass recomputes the two up-projections and reuses the attention output the forward pass kept.
    """
    und_norm = self.input_layernorm(und_seq)
    gen_norm = self.input_layernorm_moe_gen(gen_seq)
    und_attn, gen_attn = self.self_attn(und_norm, gen_norm, rotary_emb)
    residual_und = und_seq + und_attn
    residual_gen = gen_seq + gen_attn

    def run_mlp(residual_und: torch.Tensor, residual_gen: torch.Tensor):
        mlp_und = self.mlp(self.post_attention_layernorm(residual_und))
        mlp_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))
        return mlp_und, mlp_gen

    # `use_reentrant=False` because the recomputation has to see the autograd graph the forward built, which is what
    # lets this sit inside a layer that is itself already running under an outer backward pass.
    mlp_und, mlp_gen = checkpoint(run_mlp, residual_und, residual_gen, use_reentrant=False)
    return residual_und + mlp_und, residual_gen + mlp_gen


# Dispatched op names that carry the attention output for the backends this policy runs on: PyTorch's native packed
# CUDA kernel, Ascend's fused attention, and the SDPA reference used by the CPU test. Anything else is recomputed.
_ATTENTION_OP_PATTERNS = ("varlen_attn", "fmha", "npu_fusion_attention", "scaled_dot_product_attention")


def _is_attention_op(func) -> bool:
    """Whether a selective-checkpoint policy op is one whose output is worth keeping."""
    name = getattr(func, "__name__", "")
    return any(pattern in name for pattern in _ATTENTION_OP_PATTERNS)


class _SelectiveBudget:
    """Pick the selective-checkpoint path from the packed batch and a memory budget.

    The old path keeps the attention half live, which retains the whole attention autograd graph and costs roughly a
    few times one residual stream per layer. It is the faster backward, but at 32 samples it exceeds a 96 GiB card.
    The official-style path checkpoints the whole layer and saves only the attention op outputs up to a byte budget,
    recomputing both halves for the remaining layers. The first forward of a run decides between them and, for the
    op-level path, how many layers may keep their attention output.
    """

    def __init__(self, budget_gib: float, num_layers: int) -> None:
        self.budget_bytes = max(0.0, float(budget_gib)) * 2**30
        self.num_layers = int(num_layers)
        self.path: Optional[str] = None
        self.max_saved_layers = 0

    def decide(self, und_seq: torch.Tensor, gen_seq: torch.Tensor) -> None:
        """Decide once per run, from the first packed batch's token count."""
        if self.path is not None:
            return
        tokens = int(und_seq.shape[0]) + int(gen_seq.shape[0])
        hidden = int(und_seq.shape[-1])
        # bf16 attention output for both pathways; the MLP-only path additionally retains the attention graph, so its
        # per-layer cost is estimated at 3x the residual stream.
        residual_bytes = max(tokens * hidden * 2, 1)
        mlp_only_bytes = self.num_layers * residual_bytes * 3
        if mlp_only_bytes <= self.budget_bytes:
            self.path = "mlp"
            logger = get_dist_logger()
            logger.info(
                f"[selective-ac] tokens={tokens} hidden={hidden}: MLP-only path "
                f"(estimated {mlp_only_bytes / 2**30:.1f} GiB <= budget {self.budget_bytes / 2**30:.1f} GiB)",
                ranks=[0],
            )
        else:
            self.path = "sac"
            self.max_saved_layers = min(self.num_layers, int(self.budget_bytes / residual_bytes))
            logger = get_dist_logger()
            logger.info(
                f"[selective-ac] tokens={tokens} hidden={hidden}: SAC path, keeping attention outputs for "
                f"{self.max_saved_layers}/{self.num_layers} layers within {self.budget_bytes / 2**30:.1f} GiB",
                ranks=[0],
            )


def _sac_checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb, budget: _SelectiveBudget):
    """Official-style SAC: checkpoint the whole layer and keep only the attention op outputs within budget."""
    layer_index = self._colossal_layer_index
    original_forward = self._colossal_original_forward

    def run_layer(und, gen, cos_und, sin_und, cos_gen, sin_gen):
        return original_forward(und, gen, (cos_und, sin_und, cos_gen, sin_gen))

    def policy(ctx, func, *args, **kwargs):
        # The decision is per layer, not per call: both the causal AR and the full DM attention calls in a selected
        # layer are kept, and the remaining layers recompute both.
        if layer_index < budget.max_saved_layers and _is_attention_op(func):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return checkpoint(
        run_layer,
        und_seq,
        gen_seq,
        *rotary_emb,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
    )


def _selective_decoder_forward(self, und_seq, gen_seq, rotary_emb):
    """Dispatch to the MLP-only path for small packs and to budgeted SAC for large ones."""
    budget = self._colossal_selective_budget
    budget.decide(und_seq, gen_seq)
    if budget.path == "mlp":
        return _checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb)
    return _sac_checkpointed_decoder_forward(self, und_seq, gen_seq, rotary_emb, budget)


def enable_cosmos3_selective_checkpointing(model: torch.nn.Module, budget_gib: float = 8.0) -> int:
    """Checkpoint Cosmos 3 decoder layers, adapting the saved set to the packed batch and a memory budget.

    Small packs keep the original MLP-only path: the attention output is kept and only the two MLP blocks are
    recomputed, which is the cheaper backward. When that path's estimated activation footprint would exceed
    ``budget_gib``, the layers switch to the official-style SAC path: each layer is checkpointed whole and only the
    attention op outputs of the first ``budget_gib / per-layer-output`` layers are kept, so the remaining layers
    recompute attention too. This is what lets a 32-sample rank batch fit a 96 GiB card while still saving the
    attention recompute on as many layers as the budget allows.

    The decision is made on the first forward of the run, from the actual packed token count; it is stable for the
    rest of the run. ``budget_gib=0`` forces the SAC path with nothing saved, i.e. full recompute.

    Args:
        model (torch.nn.Module): The Cosmos 3 transformer, wrapped or unwrapped.
        budget_gib (float, optional): Memory budget for saved attention outputs (and, on the MLP-only path, the
            attention graph). Defaults to 8.0 GiB.

    Returns:
        int: Number of decoder layers patched; zero when they all were already.

    Raises:
        RuntimeError: If the model has no Cosmos 3 decoder layer, i.e. the patch would silently disable
            checkpointing and the step would run out of memory instead.
    """
    layers = [module for module in model.modules() if module.__class__.__name__ == _DECODER_LAYER_CLS]
    if not layers:
        raise RuntimeError(
            f"No {_DECODER_LAYER_CLS} module found in {type(model).__name__}; selective checkpointing only applies "
            "to a Cosmos 3 omni transformer."
        )

    budget = _SelectiveBudget(budget_gib, len(layers))
    patched = 0
    for layer_index, module in enumerate(layers):
        if getattr(module, "_colossal_selective_checkpointing", False):
            continue
        module._colossal_original_forward = module.forward
        module._colossal_selective_budget = budget
        module._colossal_layer_index = layer_index
        module.forward = MethodType(_selective_decoder_forward, module)
        module._colossal_selective_checkpointing = True
        patched += 1
    return patched
