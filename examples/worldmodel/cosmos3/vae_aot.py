"""AOTInductor compilation for the chunked Wan encoder.

The official Cosmos tokenizer compiles its encoder chunk function ahead of time and validates every variant
against eager before installing it (report Sec. 5.2.6, and ``compile_encode`` in the framework's
``wan2pt2_vae_4x16x16.py``). This module does the same for the diffusers encoder this example runs:

* one exported graph per (batch, chunk length, spatial shape, cache signature);
* ``torch.export`` + ``torch._inductor.aoti_compile_and_package`` with ``emulate_precision_casts``, the setting the
  framework added to keep Inductor's cast points close to eager;
* a correctness gate over the latent and **every returned cache tensor**, because a wrong cache silently corrupts
  the following chunk even when the current latent looks fine;
* dispatch from the chunk loop, with the eager call as the fallback for shapes that fail to compile or validate.

The compile happens on the first chunk that needs a variant, so a run pays the cost in its first step and reuses
the graph afterwards. The cache is a plain Python list containing ``None`` or tensors; a variant's graph takes the
non-``None`` entries as arguments and returns the non-``None`` entries after the call, so the ``None`` pattern of
each variant is fixed at export time.

Example:
    .. code-block:: python

        from vae_aot import AOTChunkRunner

        runner = AOTChunkRunner(vae.encoder, save_dir="/path/to/aot_vae", verbose=True)
        runner(x_chunk, vae._enc_feat_map)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch

__all__ = ["AOTChunkRunner"]

# The framework's gate: latent and every non-None cache slot, relative and absolute, after casting to fp32.
_GATE_RTOL = 1e-3
_GATE_ATOL = 0.2


def _cache_signature(cache: Sequence[Optional[torch.Tensor]]) -> Tuple[int, ...]:
    """Per-slot temporal extent of the cache, ``-1`` for a ``None`` slot."""
    return tuple(-1 if entry is None else int(entry.shape[2]) for entry in cache)


def _variant_key(
    x_chunk: torch.Tensor, cache: Sequence[Optional[torch.Tensor]]
) -> Tuple[int, int, int, int, Tuple[int, ...]]:
    """Everything a compiled graph is specialised to: batch, chunk length, spatial shape, cache pattern."""
    return (
        int(x_chunk.shape[0]),
        int(x_chunk.shape[2]),
        int(x_chunk.shape[3]),
        int(x_chunk.shape[4]),
        _cache_signature(cache),
    )


def _clone_cache(cache: Sequence[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
    return [None if entry is None else entry.detach().clone() for entry in cache]


def _eager_chunk(encoder, x_chunk: torch.Tensor, cache: Sequence[Optional[torch.Tensor]]):
    """Run the eager encoder on copies, returning its output and the cache it produced."""
    local_cache = _clone_cache(cache)
    with torch.no_grad():
        out = encoder(x_chunk.detach().clone(), feat_cache=local_cache, feat_idx=[0])
    return out, local_cache


class _ChunkForAOT(torch.nn.Module):
    """Exportable wrapper around one encoder chunk call.

    Args:
        encoder: The diffusers ``WanEncoder3d`` (possibly monkey-patched by the fast-encoder helpers).
        none_mask: ``True`` for every cache slot that is ``None`` in this variant's input.
    """

    def __init__(self, encoder, none_mask: Sequence[bool]) -> None:
        super().__init__()
        self.encoder = encoder
        self.none_mask = tuple(bool(flag) for flag in none_mask)

    def forward(self, x_chunk: torch.Tensor, *present: torch.Tensor):
        cache: List[Optional[torch.Tensor]] = []
        iterator = iter(present)
        for is_none in self.none_mask:
            cache.append(None if is_none else next(iterator))
        out = self.encoder(x_chunk, feat_cache=cache, feat_idx=[0])
        return (out, *[entry for entry in cache if entry is not None])


class AOTChunkRunner:
    """Compile and dispatch the encoder's per-chunk call.

    Args:
        encoder: The diffusers ``WanEncoder3d``.
        save_dir: Directory for the ``.pt2`` packages; created on first use.
        verbose (bool, optional): Print one line per compile, gate result and fallback. Defaults to False.

    The runner is callable as ``runner(x_chunk, cache)`` and mutates ``cache`` in place, matching
    ``encoder(x_chunk, feat_cache=cache, feat_idx=[0])``.
    """

    def __init__(self, encoder, save_dir: Optional[Path] = None, verbose: bool = False) -> None:
        self.encoder = encoder
        self.save_dir = Path(save_dir) if save_dir is not None else Path("aot_vae")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.functions: dict[Tuple, Callable] = {}
        self.output_none_masks: dict[Tuple, Tuple[bool, ...]] = {}

    def __call__(self, x_chunk: torch.Tensor, cache: List[Optional[torch.Tensor]]) -> torch.Tensor:
        key = _variant_key(x_chunk, cache)
        function = self.functions.get(key)
        if function is None:
            function = self._compile(key, x_chunk, cache)
            if function is None:
                with torch.no_grad():
                    return self.encoder(x_chunk, feat_cache=cache, feat_idx=[0])
            self.functions[key] = function

        present = [entry for entry in cache if entry is not None]
        result = function(x_chunk, *present)
        out = result[0]
        returned = iter(result[1:])
        output_none_mask = self.output_none_masks[key]
        for index, is_none in enumerate(output_none_mask):
            if is_none:
                cache[index] = None
            else:
                # Copy into a fresh contiguous tensor: AOTInductor may return a view of a workspace buffer, which
                # the next variant's graph must not alias (it is specialised on the layout of eager cache tensors).
                cache[index] = next(returned).detach().clone(memory_format=torch.contiguous_format)
        return out

    def _compile(self, key: Tuple, x_chunk: torch.Tensor, cache: Sequence[Optional[torch.Tensor]]):
        eager_out, eager_cache = _eager_chunk(self.encoder, x_chunk, cache)
        output_none_mask = tuple(entry is None for entry in eager_cache)

        none_mask = tuple(entry is None for entry in cache)
        present = [entry.detach() for entry in cache if entry is not None]
        wrapper = _ChunkForAOT(self.encoder, none_mask)
        package = (
            self.save_dir / f"chunk_{hashlib.sha256(repr((key, str(x_chunk.dtype))).encode()).hexdigest()[:16]}.pt2"
        )

        try:
            with torch.no_grad():
                exported = torch.export.export(wrapper, (x_chunk.detach(), *present), strict=False)
                torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=str(package),
                    inductor_configs={
                        "aot_inductor.package_constants_in_so": False,
                        # Same setting as the framework's VAE fix: keep Inductor's low-precision casts close to
                        # eager so multi-frame caches pass the gate below.
                        "emulate_precision_casts": True,
                        # The framework's other workaround for the same miscompile: comprehensive padding hands the
                        # AvgDown3D shortcut a padded buffer whose generated index uses a different stride order.
                        # Disabling it is the validated alternative to removing the identity view path.
                        "comprehensive_padding": False,
                    },
                )
            function = torch._inductor.aoti_load_package(str(package), device_index=torch.cuda.current_device())
            state = wrapper.state_dict()
            required = set(function.get_constant_fqns())
            function.load_constants(
                {name: value for name, value in state.items() if name in required},
                check_full_update=True,
                user_managed=True,
            )
        except Exception as error:  # noqa: BLE001 - any compile/load failure must fall back to eager
            if self.verbose:
                print(f"[vae-aot] compile failed for {key}: {error}", flush=True)
            return None

        if not self._matches_eager(function, x_chunk, cache, eager_out, eager_cache, output_none_mask):
            if self.verbose:
                print(f"[vae-aot] variant failed the correctness gate: {key}", flush=True)
            return None

        self.output_none_masks[key] = output_none_mask
        if self.verbose:
            print(f"[vae-aot] installed {key} ({package.name})", flush=True)
        return function

    def _matches_eager(self, function, x_chunk, cache, eager_out, eager_cache, output_none_mask) -> bool:
        """Compare the latent and every returned cache slot with the eager result."""
        present = [entry.detach() for entry in cache if entry is not None]
        try:
            with torch.no_grad():
                result = function(x_chunk.detach(), *present)
        except Exception:  # noqa: BLE001
            return False
        aot_out = result[0]
        returned = iter(result[1:])
        aot_cache: List[Optional[torch.Tensor]] = []
        for is_none in output_none_mask:
            aot_cache.append(None if is_none else next(returned))

        def close(aot_tensor, eager_tensor) -> bool:
            if aot_tensor is None or eager_tensor is None:
                return aot_tensor is None and eager_tensor is None
            return torch.allclose(aot_tensor.float(), eager_tensor.float(), rtol=_GATE_RTOL, atol=_GATE_ATOL)

        if not close(aot_out, eager_out):
            return False
        for aot_entry, eager_entry in zip(aot_cache, eager_cache):
            if not close(aot_entry, eager_entry):
                return False
        return True
