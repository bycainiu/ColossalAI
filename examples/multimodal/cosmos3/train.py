r"""Minimal ColossalAI training entry point for Cosmos 3.

The script is deliberately small: it owns the distributed launch, the plugin/policy wiring, metadata lifetime,
token-normalised loss reduction, checkpoint state and held-out evaluation. The data recipe (how a sample's latents,
targets, timesteps and loss masks are produced) stays outside; a replay pack is a ``.pt`` file holding a list of
packer-shaped sample dictionaries, exactly the input of
:func:`colossalai.shardformer.modeling.cosmos_mot.pack_cosmos3_samples`.

Only the ``BF16 + DP/ZeRO + tp=pp=sp=1`` path is in scope. Tensor, pipeline and sequence parallelism are not
validated for Cosmos 3, and the policy rejects PP/SP.

Run a CPU-only contract check (no diffusers, no distributed):

    python examples/multimodal/cosmos3/train.py --dry-run

Launch a real run on two H20 GPUs:

    torchrun --nproc_per_node=2 examples/multimodal/cosmos3/train.py \
        --model /path/to/Cosmos3-Edge --replay-pack /path/to/replay-packs.pt \
        --backend cuda --zero-stage 2 --steps 100 --save-dir /path/to/run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from colossalai.shardformer.modeling.cosmos3 import clear_cosmos3_packed_meta, set_cosmos3_packed_meta
from colossalai.shardformer.modeling.cosmos_mot import Cosmos3PackedBatch, pack_cosmos3_samples, two_way_varlen_summary

try:
    from examples.multimodal.cosmos3.loss import combine_modality_losses, normalise_packed_loss, velocity_mse
    from examples.multimodal.cosmos3.optim import build_adamw
except ImportError:  # direct execution from the example directory
    from loss import combine_modality_losses, normalise_packed_loss, velocity_mse
    from optim import build_adamw


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Diffusers Cosmos 3 checkpoint directory or Hub id.")
    parser.add_argument("--replay-pack", help=".pt file with a list of packer-shaped sample dictionaries.")
    parser.add_argument("--eval-pack", help="Optional .pt file used as the fixed held-out pack.")
    parser.add_argument("--heldout-manifest", help="JSON manifest produced by heldout.py; checked for overlap.")
    parser.add_argument("--backend", default="auto", choices=("auto", "naive", "npu", "cuda"))
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--zero-stage", type=int, default=2, choices=(0, 1, 2))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--adamw",
        default="auto",
        choices=("auto", "plain", "foreach", "fused"),
        help="AdamW implementation; the four arms share the same update math and exist for A/B measurement.",
    )
    parser.add_argument("--save-dir", help="Directory for model/optimizer checkpoints and training state.")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--resume", help="Checkpoint directory written by this script.")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--bench-json",
        help="Write per-step timings, after the first --bench-warmup steps, to this JSON file; the report uses the "
        "same environment block and metric definitions as bench_ops.py.",
    )
    parser.add_argument("--bench-warmup", type=int, default=5, help="Steps excluded from --bench-json timings.")
    parser.add_argument("--dry-run", action="store_true", help="Exercise packing and loss on CPU; no model or GPU.")
    return parser.parse_args(argv)


def resolve_backend(requested: str) -> str:
    """Resolve ``auto`` from the accelerator without hiding the choice in the policy default."""
    if requested != "auto":
        return requested
    from colossalai.accelerator import get_accelerator

    name = get_accelerator().name
    return {"npu": "npu", "cuda": "cuda"}.get(name, "naive")


def _build_optimizer(
    model,
    learning_rate: float,
    backend: str,
    mode: str = "auto",
    *,
    betas: Sequence[float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
):
    """Build the AdamW the training step uses; the arms and their trade-offs live in :mod:`optim`."""
    return build_adamw(
        model.parameters(),
        learning_rate,
        backend,
        mode,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def _synthetic_sample(und_len: int, dm_len: int, seed: int) -> Dict:
    """A packer-shaped sample with the fields the contract understands; used only by ``--dry-run``."""
    generator = torch.Generator().manual_seed(seed)
    sequence_length = und_len + dm_len
    return {
        "sample_id": f"synthetic-{seed}",
        "und_len": und_len,
        "sequence_length": sequence_length,
        "input_ids": torch.randint(0, 64, (und_len,), generator=generator),
        "text_indexes": torch.arange(und_len),
        "position_ids": torch.arange(sequence_length).unsqueeze(0).repeat(3, 1),
        "vision_sequence_indexes": torch.arange(und_len, sequence_length),
        "vision_mse_loss_indexes": torch.arange(und_len, sequence_length),
        "vision_timesteps": torch.full((dm_len,), 500.0),
        "vision_tokens": [torch.randn(1, 4, 2, 4, 4, generator=generator)],
        "vision_token_shapes": [(2, 2, 2)],
        "vision_noisy_frame_indexes": [torch.arange(1)],
        "vision_domain_ids": [torch.tensor(0)],
    }


def dry_run(args: argparse.Namespace) -> int:
    """Check the batch contract and the loss reduction without importing diffusers or initialising a process group."""
    samples = [_synthetic_sample(12, 48, seed=1), _synthetic_sample(8, 40, seed=2)]
    batch = pack_cosmos3_samples(samples, pad_tokens=4)
    prediction = torch.randn(batch.total_supervised_tokens)
    target = prediction + 0.01
    loss_sum, token_count = velocity_mse(prediction, target)
    combined = combine_modality_losses([{"sum": loss_sum, "count": token_count}])
    loss = normalise_packed_loss(combined["sum"], combined["count"], combined["count"], 1)

    report = {
        "status": "dry-run",
        "sample_ids": list(batch.sample_ids),
        "packed_tokens": batch.layout.sequence_length,
        "real_tokens": batch.layout.real_tokens,
        "pad_tokens": batch.layout.pad_tokens,
        "ar_dm_lengths": {"ar": batch.meta.ar_lens, "dm": batch.meta.dm_lens},
        "supervised_tokens": batch.total_supervised_tokens,
        "loss": float(loss.detach().cpu()),
        "varlen_summary": two_way_varlen_summary(batch.meta),
        "backend": resolve_backend(args.backend),
    }
    print(json.dumps(report, indent=2))
    if batch.total_supervised_tokens == 0:
        raise SystemExit("dry-run produced no supervised token")
    if not torch.isfinite(loss).item():
        raise SystemExit("dry-run produced a non-finite loss")
    return 0


def _load_replay_packs(path: str | Path) -> List[List[Dict]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} must hold a non-empty list of packs (each pack is a list of sample dictionaries).")
    if isinstance(payload[0], dict):
        return [payload]
    return payload


def _to_device(batch: Cosmos3PackedBatch, device: torch.device) -> Cosmos3PackedBatch:
    moved = {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.model_inputs.items()
    }
    return Cosmos3PackedBatch(
        model_inputs=moved,
        layout=batch.layout,
        sample_ids=batch.sample_ids,
        supervised_tokens=batch.supervised_tokens,
    )


def _loss_for_batch(
    model,
    batch: Cosmos3PackedBatch,
    samples: Sequence[Dict],
    data_parallel_size: int,
) -> torch.Tensor:
    """Compute the token-normalised velocity loss for one packed batch.

    The replay pack must carry ``vision_target`` per sample, a list matching the model's ``sample`` output. A full
    recipe replaces this with its noise/timestep and modality weights; the reduction and metadata binding stay the
    same.
    """
    device = next(model.parameters()).device
    set_cosmos3_packed_meta(model, batch.meta, strict=True)
    outputs = model(**_to_device(batch, device).model_inputs)
    modality_losses = []
    for sample_index, sample in enumerate(samples):
        target = sample["vision_target"].to(device)
        prediction = outputs.sample[sample_index]
        modality_losses.append(velocity_mse(prediction, target))

    combined = combine_modality_losses([{"sum": sum_loss, "count": count} for sum_loss, count in modality_losses])
    local_tokens = int(combined["count"])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from colossalai.shardformer.modeling.cosmos_mot import all_reduce_supervised_tokens

        global_tokens = all_reduce_supervised_tokens(local_tokens, device=device)
    else:
        global_tokens = local_tokens
    return normalise_packed_loss(combined["sum"], local_tokens, global_tokens, data_parallel_size)


def _device_rng_state():
    """Return the accelerator RNG state, or None when the current accelerator does not expose one (CPU)."""
    from colossalai.accelerator import get_accelerator

    try:
        return [state.cpu() for state in get_accelerator().get_rng_state_all()]
    except (RuntimeError, AttributeError):
        return None


def _save_state(directory: Path, model, optimizer, step: int, epoch: int, sampler_cursor: int) -> None:
    import torch.distributed as dist

    directory.mkdir(parents=True, exist_ok=True)
    from colossalai.booster import Booster

    # The booster object is passed through the module attribute set in `train`; saving through it keeps the
    # checkpoint format consistent with the plugin (sharded model, ZeRO optimizer).
    booster: Booster = model._colossal_booster
    booster.save_model(model, str(directory / "model"), shard=True, use_safetensors=True)
    booster.save_optimizer(optimizer, str(directory / "optimizer"), shard=True)
    state = {
        "step": step,
        "epoch": epoch,
        "sampler_cursor": sampler_cursor,
        "torch_rng_state": torch.get_rng_state(),
        "device_rng_state": _device_rng_state(),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
    }
    torch.save(state, directory / "training_state.pt")


def _restore_state(directory: Path, model, optimizer) -> Dict:
    import torch.distributed as dist

    from colossalai.accelerator import get_accelerator

    booster = model._colossal_booster
    booster.load_model(model, str(directory / "model"))
    booster.load_optimizer(optimizer, str(directory / "optimizer"))
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch_rng_state"])
    if state.get("device_rng_state") is not None:
        try:
            get_accelerator().set_rng_state_all(state["device_rng_state"])
        except (RuntimeError, AttributeError):
            # A checkpoint moved from an accelerator host to a CPU-only host has no device RNG to restore.
            pass
    if dist.is_initialized() and state.get("world_size") != dist.get_world_size():
        print(
            f"warning: resuming a {state.get('world_size')}-rank checkpoint on {dist.get_world_size()} ranks; the "
            f"model/optimizer shards are reloaded, but step-for-step equivalence is only promised for the same "
            f"topology.",
            flush=True,
        )
    return state


def _write_bench_report(
    path: Path,
    args: argparse.Namespace,
    step_times: Sequence[float],
    world_size: int,
    backend: str,
    device: torch.device,
) -> None:
    """Write the step-time report; ``bench_ops.py`` owns the environment block so both reports are comparable."""
    timings = list(step_times)[args.bench_warmup :]
    if not timings:
        raise SystemExit("--bench-json needs more steps than --bench-warmup")
    try:
        from examples.multimodal.cosmos3.bench_ops import collect_env
    except ImportError:  # direct execution from the example directory
        from bench_ops import collect_env

    ordered = sorted(timings)
    report = {
        "mode": "train-step",
        "env": {**collect_env(device), "world_size": world_size, "backend": backend},
        "conditions": {
            "steps": len(timings),
            "warmup_steps": args.bench_warmup,
            "precision": args.precision,
            "zero_stage": args.zero_stage,
            "adamw": args.adamw,
            "learning_rate": args.learning_rate,
        },
        "step_ms": {
            "median": ordered[len(ordered) // 2] * 1e3,
            "min": ordered[0] * 1e3,
            "mean": sum(timings) / len(timings) * 1e3,
        },
        "steps_ms": [value * 1e3 for value in timings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> int:
    if not args.model or not args.replay_pack:
        raise SystemExit("a real run needs --model and --replay-pack; use --dry-run for the CPU contract check.")
    if not args.save_dir:
        raise SystemExit("a real run needs --save-dir so the checkpoint chain is explicit.")

    import torch.distributed as dist

    import colossalai
    from colossalai.booster import Booster
    from colossalai.booster.plugin import HybridParallelPlugin
    from colossalai.shardformer.policies.cosmos3 import Cosmos3OmniPolicy

    colossalai.launch_from_torch(config={})
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    backend = resolve_backend(args.backend)

    diffusers = __import__("diffusers")
    transformer_cls = getattr(diffusers, "Cosmos3OmniTransformer", None)
    if transformer_cls is None:
        raise SystemExit(
            f"diffusers {getattr(diffusers, '__version__', 'unknown')} does not expose Cosmos3OmniTransformer. "
            f"Install the pinned Cosmos 3 Diffusers source recorded in docs/HF5-COMPATIBILITY.md."
        )
    model = transformer_cls.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    policy = Cosmos3OmniPolicy(backend=backend)
    plugin = HybridParallelPlugin(
        tp_size=1,
        pp_size=1,
        sp_size=1,
        precision=args.precision,
        zero_stage=args.zero_stage,
        custom_policy=policy,
        overlap_communication=True,
        max_norm=0.0,
    )
    optimizer = _build_optimizer(model, args.learning_rate, backend, args.adamw)
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
    model._colossal_booster = booster

    packs = _load_replay_packs(args.replay_pack)
    start_step = 0
    epoch = 0
    sampler_cursor = 0
    if args.resume:
        state = _restore_state(Path(args.resume), model, optimizer)
        start_step = int(state["step"])
        epoch = int(state["epoch"])
        sampler_cursor = int(state["sampler_cursor"])

    data_parallel_size = world_size
    begin = time.time()
    step_times: List[float] = []
    for step in range(start_step, args.steps):
        step_start = time.perf_counter()
        samples = packs[(sampler_cursor + step) % len(packs)]
        batch = pack_cosmos3_samples(samples)
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_for_batch(model, batch, samples, data_parallel_size)
        booster.backward(loss, optimizer)
        optimizer.step()
        clear_cosmos3_packed_meta(model, strict=True)
        step_times.append(time.perf_counter() - step_start)

        if rank == 0 and (step + 1) % args.log_interval == 0:
            elapsed = time.time() - begin
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach().cpu()),
                        "supervised_tokens": batch.total_supervised_tokens,
                        "real_tokens": batch.layout.real_tokens,
                        "pad_tokens": batch.layout.pad_tokens,
                        "samples_per_second": len(samples) * (step + 1 - start_step) / max(elapsed, 1e-9),
                    }
                ),
                flush=True,
            )
        if args.save_dir and (step + 1) % args.save_interval == 0:
            _save_state(Path(args.save_dir), model, optimizer, step + 1, epoch, sampler_cursor + step + 1)

    if args.bench_json and rank == 0:
        _write_bench_report(
            Path(args.bench_json),
            args,
            step_times,
            world_size,
            backend,
            next(model.parameters()).device,
        )

    _save_state(Path(args.save_dir), model, optimizer, args.steps, epoch, sampler_cursor + args.steps)

    if args.eval_pack:
        eval_packs = _load_replay_packs(args.eval_pack)
        with torch.no_grad():
            losses = []
            for samples in eval_packs:
                batch = pack_cosmos3_samples(samples)
                losses.append(float(_loss_for_batch(model, batch, samples, data_parallel_size).detach().cpu()))
                clear_cosmos3_packed_meta(model, strict=True)
        if rank == 0:
            print(json.dumps({"heldout_losses": losses}), flush=True)

    dist.destroy_process_group()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        return dry_run(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
