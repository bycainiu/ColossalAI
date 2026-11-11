"""SFT for Cosmos3-Edge-Policy-DROID with ColossalAI on Ascend or CUDA.

Follows the ``action_policy_droid_nano`` recipe of nvidia/cosmos-framework, with Edge's model config. The device is
read off ``get_accelerator()``; ``--backend`` selects the two-way attention kernel that matches it (``npu`` on
Ascend, ``cuda`` on NVIDIA).

Launch with::

    python -m torch.distributed.run --nproc_per_node=8 train_sft.py \
        --data /path/to/Cosmos3-DROID/success \
        --checkpoint /path/to/Cosmos3-Edge-Policy-DROID \
        --out /path/to/runs/cosmos3-sft

``torchrun`` from a virtualenv that inherits torch from the system interpreter may resolve to that interpreter
instead, which will not have ColossalAI importable; ``python -m torch.distributed.run`` avoids that.

To exercise the training loop without video or the VAE, point ``--latents`` at a safetensors file of
precomputed latents (see ``LatentPolicyDataset``).

The ``__main__`` guard is required: the DataLoader starts its workers with ``spawn``, which re-imports this
module in every worker, so any top-level side effect would run again there.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from droid_dataset import DroidActionPolicyDataset, collate
from packing import compute_loss, pack_batch
from torch.utils.data import Dataset

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

try:
    from examples.worldmodel.cosmos3_harness.training_metrics import (
        AnomalyDetector,
        ConsoleObserver,
        Stopwatch,
        TrainingMetrics,
        WandbObserver,
        device_memory_gib,
    )
except ImportError:  # direct execution from the example directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cosmos3_harness"))
    from training_metrics import (
        AnomalyDetector,
        ConsoleObserver,
        Stopwatch,
        TrainingMetrics,
        WandbObserver,
        device_memory_gib,
    )

# --- Recipe constants, from action_policy_droid_nano. Not exposed as flags on purpose. ------------------------
LR = 2.0e-04  # paired with a global batch of 8192
BETAS = (0.9, 0.99)
EPS = 1.0e-08  # the recipe notes that bf16 parameters with eps 1e-6 make the action loss diverge
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0

# Train the generation tower and the action heads; the AR tower stays frozen. These are the diffusers names; the
# cosmos-framework "moe_gen" parameters appear in diffusers as the *_moe_gen modules plus the add_* attention
# projections.
TRAINABLE_KEYS = (
    "_moe_gen",  # mlp_moe_gen / input_layernorm_moe_gen / post_attention_layernorm_moe_gen
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "norm_added_q",
    "norm_added_k",
    "k_norm_und_for_gen",
    "time_embedder",
    "proj_in",
    "proj_out",  # vae2llm / llm2vae
    "action_proj_in",
    "action_proj_out",  # action2llm / llm2action
    "action_modality_embed",
)
# These get 5x the base learning rate (the recipe's lr_multipliers).
LR_MULTIPLIED = ("action_proj_in", "action_proj_out", "action_modality_embed")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="DROID dataset root, in LeRobot v3 layout.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Cosmos3-Edge-Policy-DROID directory.")
    parser.add_argument("--out", type=Path, default=Path("runs/cosmos3-sft"), help="Checkpoint output directory.")
    parser.add_argument(
        "--latents",
        type=Path,
        help="Skip the VAE and read precomputed latents from this safetensors file or directory.",
    )
    parser.add_argument(
        "--latent-task",
        default="pick up the object and put it in the container",
        help="Task text for latent files whose metadata does not carry one.",
    )

    # Recipe values by default; overridable for smoke runs at a shorter window or a smaller batch.
    parser.add_argument("--chunk-length", type=int, default=32, help="Actions per window; frames are this plus 1.")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--resolution", default="480")
    parser.add_argument(
        "--samples-per-rank",
        type=int,
        default=32,
        help="8 ranks x 32 samples x 32 accumulation steps = 8192, the global batch of the 256-rank recipe.",
    )
    parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument(
        "--sp-size",
        type=int,
        default=1,
        help="Ulysses context-parallel degree: split every packed sequence across this many ranks inside the data "
        "parallel group. Raises the sequence length a step can hold, not the sample count.",
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=None,
        help="Hybrid sharded data parallelism: ZeRO shards, reduce-scatters and all-gathers stay inside a group of "
        "this many data parallel ranks (one node), and only a shard-sized all-reduce crosses nodes. Defaults to "
        "the whole data parallel group, i.e. plain ZeRO.",
    )
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument(
        "--zero-stage",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="ZeRO stage inside the data-parallel group. The 8-rank recipe runs stage 2. At world size 1 there is no "
        "data parallel group to shard, so stage 2 only adds all-gathers and reduce-scatters; use 0 for a single-rank "
        "run.",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        help="Directory for the per-rank JSONL metrics; defaults to <out>/metrics.",
    )
    parser.add_argument("--metrics-every", type=int, default=10, help="Aggregate metrics every N optimizer steps.")
    parser.add_argument("--wandb-project", help="Log the aggregated metrics to Weights & Biases from rank 0.")
    parser.add_argument(
        "--wandb-mode",
        default="offline",
        choices=("offline", "online", "disabled"),
        help="W&B mode; offline (the default) writes the run locally and can be synced later.",
    )
    parser.add_argument("--wandb-run-name", help="W&B run name; defaults to W&B's generated name.")

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-episodes", type=int, help="Index only the first N episodes; the full index holds millions of windows."
    )
    parser.add_argument(
        "--mp-context",
        default="spawn",
        help="Decoding video in the parent before a fork deadlocks, so spawn is the default. Reading precomputed "
        "latents touches neither PyAV nor the VAE in the parent, where fork is fine.",
    )
    parser.add_argument(
        "--backend",
        default="npu",
        choices=["npu", "naive", "cuda"],
        help="Two-way attention backend. 'npu' is the Ascend two-call path, 'cuda' is PyTorch's native packed "
        "varlen kernel, and 'naive' is the PyTorch reference: it materializes the full score matrix and runs out "
        "of memory at the recipe's sequence length, so it is for numerical comparison only.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        default="full",
        choices=["full", "selective", "none"],
        help="Activation checkpointing. 'full' recomputes every decoder layer, as the recipe trains: 3.5 GiB of "
        "activation per 480p sample. 'selective' recomputes only the two MLP blocks and keeps the attention output "
        "alive, which is the official stack's choice -- a cheaper backward for more activation memory. 'none' keeps "
        "every activation, 11.3 GiB per sample, and a single 910B2 then holds two of them.",
    )
    parser.add_argument(
        "--compile-blocks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="torch.compile every decoder layer with fullgraph=True and dynamic=True, as the official stack does. Off "
        "by default: the compile has to be paid on the first step and this step is short, so it only pays off once "
        "the artifacts are cached across runs.",
    )
    parser.add_argument(
        "--async-save",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write checkpoints from background threads so the step after a save does not wait for the disk. The "
        "weights are copied to pinned host memory before the next optimizer step, so the writers always read a "
        "stable copy.",
    )
    parser.add_argument(
        "--compile-vae",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the frozen VAE encoder with static shapes, one graph per canvas -- the official stack's "
        "second tokenizer optimisation (report Sec. 5.2.6). Off on this device because both backends were measured "
        "and neither pays: `inductor` cannot lower even a plain `sum(dim=1)` without triton-ascend, and "
        "`npugraph_ex` captures in 21 s but then encodes in 16.6 s instead of 0.93 s (18x slower). See "
        "draft_cosmos3_vae_optimization_zh.md Sec. 12; fuse the RMSNorm by hand instead.",
    )
    parser.add_argument(
        "--compile-vae-backend",
        default="npugraph_ex",
        choices=["npugraph_ex", "inductor"],
        help="torch.compile backend for --compile-vae. `inductor` is the official stack's backend and needs "
        "triton-ascend, which this image does not have; `npugraph_ex` is Ascend's own ACL-graph engine (torchair), "
        "correct but measured 18x slower than eager on this encoder.",
    )
    parser.add_argument(
        "--compile-vae-aot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the per-chunk VAE encoder ahead of time with torch.export + AOTInductor, validate every "
        "variant against eager (latent and every returned cache tensor), and fall back per variant, mirroring the "
        "official tokenizer's compile_encode. Off by default; mutually exclusive with --compile-vae.",
    )
    parser.add_argument(
        "--compile-vae-aot-dir",
        type=Path,
        help="Directory for the AOTInductor .pt2 packages; defaults to <out>/aot_vae.",
    )
    parser.add_argument(
        "--vae-dtype",
        default="float32",
        choices=["float32", "float16"],
        help="Precision of the frozen VAE encoder, which is most of a step's data-side cost. fp16 encodes one sample "
        "in 0.499 s instead of 1.229 s (2.46x) on a 910B2, and its latent differs from the fp32 one by 0.58-0.64x of "
        "the bf16 quantisation pack_batch applies anyway -- measured over 64 real DROID windows spread across all "
        "eight cards. bfloat16 is deliberately not offered: at 4.7-5.0x that quantisation it is worse than the cast "
        "it would replace.",
    )
    parser.add_argument(
        "--vae-chunk-frames",
        type=int,
        default=8,
        help="Pixel frames per encoder call, after the one-frame prime. diffusers' own window is 4 (one latent "
        "chunk). The official tokenizer encodes 24 at 480p and calls this the bulk of its speedup (report Sec. "
        "5.2.6), but that is tuned for its GPUs: on a 910B2 the encoder is already device-saturated at 4 frames. "
        "Before --fast-vae-encoder, wider windows bought 1.04-1.05x by amortising the pad+cat copy; with that copy "
        "gone the window no longer trades time for memory -- 4/8/12 frames all encode in 0.70-0.72 s at batch 2, "
        "while peak grows 4.9 -> 6.6 -> 8.4 GiB -- so 8 is the knee (5 calls, 1.03x over 4, 1.7 GiB under 12). "
        "Measured in scripts/profile_vae_recipe.py. The causal caches carry across calls, so the latent is "
        "unchanged to 3.6e-05 relative, 0.03x of the bf16 cast pack_batch applies. 0 keeps diffusers' loop. Must "
        "be a multiple of 4.",
    )
    parser.add_argument(
        "--fuse-vae-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap the frozen VAE's RMSNorm for a cheaper forward: the reduction stays in fp32, but `scale / norm` "
        "is divided on the small [B,1,T,H,W] tensor instead of the activation, the dead `+0.0` bias disappears, and "
        "the fp32 up/down-cast round trip goes away -- six full passes over the activation become two. Numerically "
        "it adds at most one rounding of the input dtype per element and lands at 0.615x the bf16 cast pack_batch "
        "applies anyway (diffusers' own is 0.604x). Measured 0.927 -> 0.748 s (1.24x) in fp16 at batch 2, and "
        "1.05 GiB less peak; only 1.06x in fp32, so this default matters for the fp16 recipe. "
        "See scripts/profile_vae_norm.py and draft_cosmos3_vae_optimization_zh.md Sec. 13.",
    )
    parser.add_argument(
        "--fast-vae-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Two exact changes to the frozen VAE's convolution path: the symmetric H/W padding is handed to the "
        "convolution instead of being materialised by F.pad (in steady state the feature cache supplies the causal "
        "T frames, so no pad is copied at all), and the dead `x.clone()` in WanResidualDownBlock is dropped. Both "
        "are bit-identical to diffusers' forward -- tests/test_vae_fast_encoder.py asserts torch.equal -- so unlike "
        "the fused norm there is no accuracy trade at all. See draft_cosmos3_vae_optimization_zh.md Sec. 14.",
    )
    args = parser.parse_args()
    if args.data is None and args.latents is None:
        parser.error("one of --data or --latents is required")
    if args.vae_chunk_frames and args.vae_chunk_frames % 4:
        parser.error("--vae-chunk-frames must be a multiple of 4")
    if args.compile_vae and args.compile_vae_aot:
        parser.error("--compile-vae (JIT) and --compile-vae-aot are different mechanisms; enable one.")
    return args


def build_model(checkpoint: Path):
    """Load the transformer and freeze the AR tower. The model comes from diffusers, not HF transformers."""
    from diffusers import Cosmos3OmniTransformer

    model = Cosmos3OmniTransformer.from_pretrained(checkpoint / "transformer", torch_dtype=torch.bfloat16)
    trainable = frozen = 0
    for name, p in model.named_parameters():
        p.requires_grad = any(k in name for k in TRAINABLE_KEYS)
        if p.requires_grad:
            trainable += p.numel()
        else:
            frozen += p.numel()
    print(f"[model] trainable {trainable / 1e9:.3f}B / frozen {frozen / 1e9:.3f}B", flush=True)
    return model


def build_tokenizer(checkpoint: Path):
    """The text tower's tokenizer ships inside the checkpoint next to the transformer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint / "text_tokenizer")


def build_vae(
    checkpoint: Path,
    dtype: torch.dtype = torch.float32,
    chunk_frames: int = 0,
    fuse_norm: bool = False,
    fast_encoder: bool = True,
    aot_dir: Path = None,
):
    """Load the VAE frozen. It runs inside the training loop and never enters the booster.

    HF32 has to be off. With it on, the error of an fp32 convolution on Ascend was measured at 210x the CPU
    result, which leaves the latents subtly wrong; the symptom is a loss curve that does not match the reference
    device, which is very hard to trace back to here.

    ``dtype`` is the encoder's precision; see ``--vae-dtype``. It is passed to ``from_pretrained`` rather than
    applied with ``.to()`` afterwards, so that any module diffusers wants kept in fp32 stays fp32.

    ``chunk_frames`` widens the encoder call window past diffusers' four pixel frames; see ``--vae-chunk-frames``.
    ``fuse_norm`` and ``fast_encoder`` replace the two cheapest passes diffusers makes over the activation; see
    ``--fuse-vae-norm`` and ``--fast-vae-encoder``. ``aot_dir`` compiles and validates the per-chunk encoder with
    AOTInductor into that directory; see ``--compile-vae-aot``.
    """
    from diffusers import AutoencoderKLWan

    from colossalai.accelerator import get_accelerator

    if get_accelerator().name == "npu":
        # With HF32 on, an fp32 convolution on Ascend was measured at 210x the CPU error. On CUDA the analogue is
        # TF32, so both are disabled for the frozen encoder.
        torch.npu.conv.allow_hf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    vae = AutoencoderKLWan.from_pretrained(checkpoint / "vae", torch_dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad = False
    if chunk_frames:
        from vae_chunked import enable_chunked_encode

        enable_chunked_encode(vae, chunk_frames, aot_dir=aot_dir, verbose=aot_dir is not None)
    if fuse_norm:
        from vae_fused_norm import enable_fused_rmsnorm

        enable_fused_rmsnorm(vae)
    if fast_encoder:
        from vae_fast_encoder import enable_fast_encoder

        enable_fast_encoder(vae)
    return vae


def build_optimizer(model):
    """FusedAdam with fp32 master weights; the action heads get 5x the learning rate."""
    base, boosted = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (boosted if any(k in name for k in LR_MULTIPLIED) else base).append(p)
    return HybridAdam(
        [{"params": base, "lr": LR}, {"params": boosted, "lr": LR * 5.0}],
        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
        adamw_mode=True,
    )


# ---------------------------------------------------------------------------------------------------------------
# Precomputed-latent bypass
#
# The upstream pipeline does not precompute: augmentation happens before the VAE, so caching latents freezes the
# augmentation, and the VAE runs inside the training step -- frozen, and torch.compiled by upstream's
# `compile_tokenizer` callback. Reading latents from disk is only a way to exercise the training loop and
# reproduce numbers without video, and it does not stand in for the real data pipeline.
# ---------------------------------------------------------------------------------------------------------------
_SAFETENSOR_HANDLES: dict = {}


def _open_safetensors(path):
    """Cache the mmap handle per path.

    The handle cannot live on the Dataset: a spawn worker pickles the Dataset and a file handle is not picklable.
    Keeping it at module scope gives each process its own read-only mapping of the same file.
    """
    handle = _SAFETENSOR_HANDLES.get(path)
    if handle is None:
        from safetensors import safe_open

        handle = safe_open(path, framework="pt", device="cpu")
        _SAFETENSOR_HANDLES[path] = handle
    return handle


class LatentPolicyDataset(Dataset):
    """Reads precomputed latents and actions, skipping the VAE.

    A sample is stored as ``"<name>/latent"`` of shape ``[1, C, T, H, W]`` alongside ``"<name>/action"`` of shape
    ``[chunk, action_dim]``. Entries without an action are skipped: a latent-only fixture carries no supervision.

    The file header must also hold ``clips.<name>`` with ``canvas_hw`` and ``aspect_key``, because the JSON prompt
    needs the on-screen geometry and that is the one piece of pixel information encoding throws away. A file
    missing them raises rather than guessing.

    Args:
        sources: A safetensors file, a directory of them, or a list of either.
        default_task (str): Task text for clips whose metadata does not carry one.
    """

    def __init__(self, sources, default_task: str):
        if isinstance(sources, (str, os.PathLike)):
            sources = [sources]
        files = []
        for source in sources:
            path = Path(source)
            if path.is_dir():
                files.extend(sorted(path.glob("*.safetensors")))
            elif path.is_file():
                files.append(path)
            else:
                raise FileNotFoundError(f"No such latent file or directory: {path}")

        self.items = []
        skipped_clips = []
        skipped_files = []
        for path in files:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = list(handle.keys())
                metadata = handle.metadata() or {}
            try:
                clips = json.loads(metadata.get("header", "{}")).get("clips", {})
            except json.JSONDecodeError:  # metadata that is not JSON is treated as absent
                clips = {}
            prefixes = sorted({key.rsplit("/", 1)[0] for key in keys if key.endswith("/latent")})
            if not prefixes:
                skipped_files.append(path.name)
                continue
            for prefix in prefixes:
                if f"{prefix}/action" not in keys:
                    skipped_clips.append(f"{path.name}:{prefix}")
                    continue
                clip = clips.get(prefix) or {}
                missing = [key for key in ("canvas_hw", "aspect_key") if key not in clip]
                if missing:
                    raise KeyError(
                        f"{path.name}:{prefix} is missing {missing} in its header. A precomputed file has to "
                        "record canvas_hw, content_hw and aspect_key, because the pixel geometry is gone after "
                        "encoding and the prompt is built from it."
                    )
                content_hw = clip.get("content_hw") or clip["canvas_hw"]
                self.items.append(
                    {
                        "path": str(path),
                        "key": prefix,
                        "task": clip.get("task") or default_task,
                        "canvas_hw": tuple(int(v) for v in clip["canvas_hw"]),
                        "content_hw": tuple(int(v) for v in content_hw),
                        "aspect_key": str(clip["aspect_key"]),
                    }
                )

        if skipped_files:
            head = ", ".join(skipped_files[:5]) + (" ..." if len(skipped_files) > 5 else "")
            print(f"[data] skipped {len(skipped_files)} file(s) with no <name>/latent key: {head}", flush=True)
        if skipped_clips:
            head = ", ".join(skipped_clips[:5]) + (" ..." if len(skipped_clips) > 5 else "")
            print(f"[data] skipped {len(skipped_clips)} clip(s) with no action: {head}", flush=True)
        if not self.items:
            raise ValueError(
                f"No sample carrying both a latent and an action was found in {sources}. A latent-only fixture "
                "has no action supervision and cannot be trained on."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        handle = _open_safetensors(item["path"])
        return {
            "latent": handle.get_tensor(f"{item['key']}/latent")[0],  # [C, T, H, W]
            "action": handle.get_tensor(f"{item['key']}/action"),  # [chunk, action_dim]
            "task": item["task"],
            "canvas_hw": torch.tensor(item["canvas_hw"]),
            "content_hw": torch.tensor(item["content_hw"]),
            "aspect_key": item["aspect_key"],
        }


def latent_collate(batch):
    """Stack the tensors and keep the strings as lists.

    The keys match ``droid_dataset.collate`` so that ``pack_batch`` does not care whether a sample came from video
    or from a precomputed file; ``canvas_hw`` and ``content_hw`` have to stay tensors, since the prompt builder
    calls ``.tolist()`` on them.
    """
    return {
        "latent": torch.stack([item["latent"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "task": [item["task"] for item in batch],
        "canvas_hw": torch.stack([item["canvas_hw"] for item in batch]),
        "content_hw": torch.stack([item["content_hw"] for item in batch]),
        "aspect_key": [item["aspect_key"] for item in batch],
    }


def fix_checkpoint_modes(save_dir: Path, coord: DistCoordinator):
    """Make a finished checkpoint readable by the rest of the group.

    safetensors' save_file bypasses the umask and writes 0600, so a checkpoint saved by one user is unreadable to
    everyone else until the modes are fixed up. With an async save the files do not exist yet at the call site, which
    is why this runs only once the writers are joined.
    """
    if not coord.is_master():
        return
    for p in save_dir.rglob("*"):
        os.chmod(p, 0o664 if p.is_file() else 0o775)


def main():
    args = parse_args()

    # torch_npu registers the 'npu' device and its operators as an import side effect; no name from it is used
    # here, so it goes through importlib rather than as an import statement nothing references. On a CUDA host the
    # module does not exist, so the import is conditional on the accelerator.
    from colossalai.accelerator import get_accelerator

    accelerator = get_accelerator()
    if accelerator.name == "npu":
        importlib.import_module("torch_npu")

    # launch_from_torch defaults to backend='nccl', which Ascend cannot use. Reading it off the accelerator rather
    # than hard-coding 'hccl' keeps the same script runnable on a GPU host.
    colossalai.launch_from_torch(backend=accelerator.communication_backend)
    coord = DistCoordinator()
    device = accelerator.get_current_device()

    vae = None
    if args.latents:
        dataset = LatentPolicyDataset(args.latents, default_task=args.latent_task)
        collate_fn = latent_collate
        coord.print_on_master(f"[data] {len(dataset)} precomputed sample(s), VAE skipped")
    else:
        dataset = DroidActionPolicyDataset(
            args.data,
            chunk_length=args.chunk_length,
            fps=args.fps,
            resolution=args.resolution,
            augment=True,
            max_episodes=args.max_episodes,
        )
        collate_fn = collate
        coord.print_on_master(f"[data] {len(dataset)} window(s)")

    model = build_model(args.checkpoint)
    if args.checkpoint_mode == "full":
        # diffusers checkpoints every entry of `self.layers`; the AR tower's layers are frozen but still recomputed.
        model.enable_gradient_checkpointing()
    model_config = model.config  # booster returns a wrapper, so keep the config before boosting
    tokenizer = build_tokenizer(args.checkpoint)
    if not args.latents:
        aot_dir = None
        if args.compile_vae_aot:
            aot_dir = args.compile_vae_aot_dir or (args.out / "aot_vae")
        vae = build_vae(
            args.checkpoint,
            getattr(torch, args.vae_dtype),
            args.vae_chunk_frames,
            args.fuse_vae_norm,
            args.fast_vae_encoder,
            aot_dir=aot_dir,
        ).to(device)
        coord.print_on_master(
            f"[vae] dtype={args.vae_dtype} window={args.vae_chunk_frames or 4} frames per encoder call, "
            f"norm={'fused' if args.fuse_vae_norm else 'diffusers'}, "
            f"conv={'fast' if args.fast_vae_encoder else 'diffusers'}, "
            f"aot={'on' if aot_dir is not None else 'off'}"
        )
        if args.compile_vae:
            # Compile the private encoder, not `encode`: the public method carries diffusers' forward hook. Static
            # shapes, so there is one graph per canvas bucket rather than one graph per step -- the encoder is called
            # on a whole batch of identically shaped windows. `dynamic=True` would defeat the capture.
            vae._encode = torch.compile(vae._encode, backend=args.compile_vae_backend, dynamic=False)
    optimizer = build_optimizer(model)

    # Cosmos3OmniPolicy does three things: it installs the two-way attention processor in postprocess(), after the
    # ShardFormer module replacements so that nothing overwrites it; with tp > 1 it applies the usual column/row
    # parallel rewrites; and with sp > 1 it switches the processor to the Ulysses context-parallel one and wires the
    # model's CP seams. With `--checkpoint-mode selective` it also patches each decoder layer's forward there, so the
    # patch lands on the layers as they come out of the shard pass. At tp = sp = 1 with full checkpointing it rewrites
    # no module and only installs the processor.
    #
    # The three launch knobs map onto the report's stack (Sec. 5.2.3): `--hsdp-shard-size` is HSDP (ZeRO sharded inside
    # a node, replicated across nodes), `--sp-size` is Ulysses CP, and `--zero-stage`-style sharding is what the plugin
    # does inside the shard group. Only DP is useful at 8 cards for this workload -- the VAE is 70% of the step and is
    # per-sample inference -- but CP is what lets a step hold a longer pack, and HSDP is what keeps the 16-card run's
    # collectives inside a node.
    #
    # custom_policy is only wired into HybridParallelPlugin -- LowLevelZero and Gemini have no such argument --
    # so the plugin has to be this one, with tp and pp at 1 and only its ZeRO in use.
    policy = Cosmos3OmniPolicy(
        backend=args.backend, selective_checkpointing=args.checkpoint_mode == "selective"
    )

    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        sp_size=args.sp_size,
        enable_sequence_parallelism=args.sp_size > 1,
        sequence_parallelism_mode="all_to_all" if args.sp_size > 1 else None,
        zero_stage=args.zero_stage,
        precision="bf16",
        max_norm=GRAD_CLIP,
        custom_policy=policy,
        hsdp_shard_size=args.hsdp_shard_size,
    )
    booster = Booster(plugin=plugin)

    # One recipe video is [3, 33, 544, 736] float32, 158 MB, so a four-sample batch and the workers' prefetch queue
    # run to tens of gigabytes of shared memory. torch's default strategy passes tensors through /dev/shm, which is
    # 32 GB in this container and fills up at samples_per_rank=4 (the dataloader then dies with "unable to allocate
    # shared memory ... No space left on device" long after the batch size looked fine). The file_system strategy
    # spills the same handles to /tmp, where the disk has terabytes; override with COSMOS3_SHM_STRATEGY when /tmp is
    # the small one on some other host.
    torch.multiprocessing.set_sharing_strategy(os.environ.get("COSMOS3_SHM_STRATEGY", "file_system"))
    dataloader = plugin.prepare_dataloader(
        dataset,
        batch_size=args.samples_per_rank,
        shuffle=False,  # the shuffle lives in the index: episode order is shuffled, windows stay sequential
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        multiprocessing_context=args.mp_context if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    if len(dataloader) == 0:
        raise ValueError(
            f"The dataloader yields no batch: {len(dataset)} sample(s), samples_per_rank="
            f"{args.samples_per_rank}, drop_last=True."
        )

    model, optimizer, _, dataloader, _ = booster.boost(model, optimizer, dataloader=dataloader)

    if args.compile_blocks:
        # Compile after boosting, not before: the policy installs the two-way attention processors in postprocess(),
        # and torch.compile traces on the first call, so it has to see the final module graph. `dynamic=True` because
        # the pack length varies with the samples the loader draws.
        target = model.unwrap()
        target.layers = torch.nn.ModuleList(
            [torch.compile(layer, fullgraph=True, dynamic=True) for layer in target.layers]
        )

    model.train()
    step = 0
    async_save_dirs = []

    try:
        device_capacity_gib = accelerator.get_device_properties(device).total_memory / 2**30
    except Exception:
        device_capacity_gib = None
    aggregate_observers = [ConsoleObserver()]
    if args.wandb_project and args.wandb_mode != "disabled":
        aggregate_observers.append(WandbObserver(args.wandb_project, args.wandb_run_name, mode=args.wandb_mode))
    metrics = TrainingMetrics(
        out_dir=args.metrics_dir or (args.out / "metrics"),
        rank=coord.rank,
        world_size=coord.world_size,
        log_every=args.metrics_every,
        aggregate_observers=aggregate_observers,
        detector=AnomalyDetector(device_capacity_gib=device_capacity_gib),
    )
    metrics.start(
        {
            "args": dict(vars(args)),
            "env": {
                "torch": torch.__version__,
                "torch_npu": (
                    getattr(importlib.import_module("torch_npu"), "__version__", "unknown")
                    if accelerator.name == "npu"
                    else "n/a"
                ),
                "device": accelerator.get_device_name(device),
            },
            "world_size": coord.world_size,
            "topology": {"sp_size": args.sp_size, "hsdp_shard_size": args.hsdp_shard_size},
            "checkpoint": args.checkpoint,
            "data": args.data,
        }
    )
    watch = Stopwatch()
    last_step_end = None
    micro_window_ms = 0.0
    micro_window_count = 0
    dataloader_iter = iter(dataloader)
    for it, batch in enumerate(dataloader_iter):
        wait_ms = None if last_step_end is None else (time.perf_counter() - last_step_end) * 1e3
        iteration_start = time.perf_counter()
        will_step = (it + 1) % args.grad_accum == 0
        record_now = will_step and ((step + 1) % args.metrics_every == 0 or step + 1 == 1)
        print_now = will_step and ((step + 1) % args.log_every == 0 or step + 1 == 1)
        watch.reset(enabled=record_now)
        # pack_batch installs the per-sample boundaries on the policy itself; see its docstring for why they are
        # not handed back to be installed here.
        with watch.section("data"):
            inputs, targets = pack_batch(
                batch,
                vae,
                tokenizer,
                model_config,
                policy=policy,
                num_frames=args.chunk_length + 1,
                fps=args.fps,
                resolution=args.resolution,
                device=device,
                stopwatch=watch,
            )
        if it == 0:
            meta = policy.packed_meta
            coord.print_on_master(
                f"[pack] samples={meta.num_samples} AR={meta.total_ar_tokens} DM={meta.total_dm_tokens} "
                f"backend={policy.processor.backend}"
            )
        with watch.section("forward"):
            out = model(**inputs)
        raw_loss, components = compute_loss(out, targets, return_components=True)
        loss = raw_loss / args.grad_accum
        with watch.section("backward"):
            booster.backward(loss, optimizer)

        if will_step:
            if args.async_save:
                # The writers read a pinned host copy of the weights; this waits for that copy to finish, not for the
                # disk, so the step below is free to overwrite the weights. A no-op when nothing is being written.
                booster.checkpoint_io.synchronize()
            with watch.section("optim"):
                optimizer.step()
            optimizer.zero_grad()
            step += 1
            micro_window_ms += (time.perf_counter() - iteration_start) * 1e3
            micro_window_count += 1
            if record_now or print_now:
                # One device sync for the whole loss decomposition; the stopwatch events queued before it are
                # complete as well, so read_ms() below does not add a second one.
                stacked = torch.stack([raw_loss.detach(), *components.values()]).tolist()
                loss_value = stacked[0]
                component_values = dict(zip(components.keys(), stacked[1:]))
            if print_now:
                coord.print_on_master(
                    f"[step {step}] loss {loss_value:.6f} "
                    f"(v {component_values['vision']:.4f} a {component_values['action']:.4f})"
                )
            if record_now:
                timings = watch.read_ms()
                packed = policy.packed_meta
                get_grad_norm = getattr(optimizer, "get_grad_norm", None)
                metrics.step(
                    step,
                    data={
                        "samples": len(batch["task"]),
                        "ar_tokens": packed.total_ar_tokens,
                        "dm_tokens": packed.total_dm_tokens,
                        "wait_ms": wait_ms,
                        "pack_ms": timings.get("data"),
                        "vae_ms": timings.get("vae"),
                    },
                    loss={"total": loss_value, **component_values},
                    perf={
                        "step_ms": micro_window_ms / max(micro_window_count, 1),
                        "forward_ms": timings.get("forward"),
                        "backward_ms": timings.get("backward"),
                        "optim_ms": timings.get("optim"),
                    },
                    optim={
                        "grad_norm": get_grad_norm() if get_grad_norm is not None else None,
                        "lr": LR,
                    },
                    mem=device_memory_gib(),
                )
            micro_window_ms = 0.0
            micro_window_count = 0
            if step % args.save_every == 0:
                save_dir = args.out / f"step-{step}"
                booster.save_model(
                    model, str(save_dir), shard=True, use_safetensors=True, use_async=args.async_save
                )
                if args.async_save:
                    # The files appear only once the writers are joined, which is deferred to the end of the run.
                    async_save_dirs.append(save_dir)
                else:
                    fix_checkpoint_modes(save_dir, coord)
            if step >= args.max_iter:
                break

        last_step_end = time.perf_counter()

    # A run that stops at --max-iter leaves the prefetch iterator alive; with persistent_workers=True its worker
    # processes then outlive the loop, and the interpreter waits for them at exit while they still hold the device.
    # Shut them down explicitly so the process exits and releases the GPU.
    shutdown_workers = getattr(dataloader_iter, "_shutdown_workers", None)
    if shutdown_workers is not None:
        shutdown_workers()

    metrics.finish({"steps": step})

    if async_save_dirs:
        booster.checkpoint_io._sync_io()
        for save_dir in async_save_dirs:
            fix_checkpoint_modes(save_dir, coord)

    if os.environ.get("COSMOS3_REPORT_PEAK_HBM"):
        # Opt-in measurement knob: the caching allocator's high-water mark on this rank's device. `npu-smi`/`nvidia-smi`
        # return nothing while the cards are busy, so this is the only reliable way to compare the checkpointing modes.
        peak = device_memory_gib().get("peak_gib", 0.0)
        print(f"[mem rank {coord.rank}] peak allocated {peak:.2f} GiB", flush=True)

    coord.print_on_master("done")


if __name__ == "__main__":
    main()
