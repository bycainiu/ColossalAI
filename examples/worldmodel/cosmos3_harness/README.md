# Cosmos 3 harness: training contract, benchmarks and evaluation

This is the shared GPU/Ascend harness for Cosmos 3. The DROID action-policy SFT recipe lives next door in
[`../cosmos3`](../cosmos3/README.md) and owns the data pipeline (LeRobot decoding, the VAE, the prompt and the noise
schedule); the framework pieces live in `colossalai/shardformer/`; this directory owns the data contract, the loss
reduction, the training loop and the evaluation/benchmark manifests.

## What is supported

- One packed batch containing several samples, laid out as a global AR block followed by a global DM block.
- BF16, data parallel + ZeRO (`tp=pp=sp=1`). The policy rejects pipeline and sequence parallelism.
- `--backend naive` as the portable reference, `--backend cuda` for PyTorch's native packed `varlen_attn` kernel, and
  `--backend npu` for the Ascend two-call kernel.
- Synchronous model/optimizer checkpoints plus a separate `training_state.pt` for step, sampler cursor and RNG.
- A grouped held-out split that never puts two windows of the same trajectory on both sides.

TP/PP/SP, Super/Nano variants and the full audio/action generation recipe are outside this scope. Do not claim them
from a passing smoke run.

## Quick checks that need no accelerator or checkpoint

```bash
export PYTHONPATH=.
python examples/worldmodel/cosmos3_harness/train.py --dry-run
bash examples/worldmodel/cosmos3_harness/test_ci.sh
```

`--dry-run` packs two synthetic samples, checks every rewritten index, computes a velocity MSE and a token-normalised
loss. It does not import diffusers and does not initialise a process group. Record the runtime baseline with:

```bash
python examples/worldmodel/cosmos3_harness/environment_report.py \
    --model nvidia/Cosmos3-Edge --revision <fixed revision> \
    --output logs/cosmos3-environment.json
```

## Held-out split

Produce a JSONL where every record has `sample_id` and `trajectory_id`, then:

```python
from examples.worldmodel.cosmos3_harness.heldout import split_by_group, write_split_manifest

train_ids, validation_ids, manifest = split_by_group(records, validation_fraction=0.1, seed=0)
write_split_manifest(manifest, "heldout.json")
```

The manifest stores a fingerprint of both sample lists. Both devices must load the same file; editing one list makes
the fingerprint check fail instead of silently comparing different splits.

## Real run

`--replay-pack` is a `.pt` file holding a list of packs; each pack is a list of sample dictionaries shaped like the
packer output (`und_len`, `sequence_length`, `input_ids`, `text_indexes`, `position_ids`, the vision/action/sound
fields, and `vision_target` for this harness). The production recipe can compute `vision_target` from its own noise and
timestep sampling; the framework contract does not change.

```bash
torchrun --nproc_per_node=2 examples/worldmodel/cosmos3_harness/train.py \
    --model /path/to/Cosmos3-Edge \
    --replay-pack /path/to/replay-packs.pt \
    --eval-pack /path/to/heldout-packs.pt \
    --backend cuda --zero-stage 2 --steps 100 \
    --save-dir /path/to/run
```

Resume with `--resume /path/to/run`. The same script runs on Ascend with `--backend npu`; the launch command is the
site's HCCL launcher. A checkpoint saved on one topology is reloaded on the same topology for step-for-step
equivalence; a different world size reloads model/optimizer shards but is reported as a topology change.

## Kernel and benchmark evidence

```bash
python examples/worldmodel/cosmos3_harness/benchmark.py \
    --backend cuda --device cuda --samples 1 2 4 8 \
    --output logs/cosmos3-attention.json
python examples/worldmodel/cosmos3_harness/benchmark.py \
    --backend naive --device cpu --samples 1 2 --iterations 2 --warmup 1 \
    --output /tmp/cosmos3-reference.json
```

The JSON records the environment, shape, backend, p50/p95 latency, K/V interleave cost and the actual fused path. Run
the naive and fused cases with the same tensors and token budget before quoting a ratio. A naive OOM is recorded as an
OOM, not extrapolated.

The optimizer arms and the end-to-end step have their own harnesses. Both report the same `env` / `conditions` /
`results` JSON and take the environment block from `bench_ops.collect_env`, so their numbers are comparable:

```bash
python examples/worldmodel/cosmos3_harness/bench_ops.py --op all --device npu \
    --num-tensors 432 --numel 400000 --output logs/cosmos3-ops.json
torchrun --nproc_per_node=8 examples/worldmodel/cosmos3_harness/train.py \
    --model /path/to/Cosmos3-Edge --replay-pack /path/to/bench-packs.pt --save-dir /tmp/bench \
    --backend npu --precision bf16 --zero-stage 2 --steps 40 --bench-warmup 10 \
    --bench-json logs/cosmos3-step.json
```

`bench_ops.py` measures `plain` / `foreach` / `fused` AdamW and `HybridAdam` on a synthetic parameter set of the
chosen shard geometry, plus the unscale/clip path before and after the multi-tensor change. `--bench-json` times the
real step and drops the first `--bench-warmup` steps. A change is accepted only when the median and the minimum both
improve; the per-step samples are kept in the JSON so a bimodal run is visible.

## Reasoner evaluation

```bash
python examples/worldmodel/cosmos3_harness/evaluate_reasoner.py \
    --benchmarks MMBench-Dev HallusionBench \
    --model-revision <pinned revision> \
    --manifest logs/cosmos3-reasoner-manifest.json
```

The manifest pins the VLMEvalKit dataset name, split, language, decoding and scorer. The actual evaluation runs in the
official VLMEvalKit environment; this directory only makes the GPU and Ascend runs share one manifest and one
per-sample response format. A subset of the General category is not a reproduction of `General Avg 59.7`.

## Testing

`test_ci.sh` runs the CPU-safe checks: packing/index rewriting, held-out fingerprinting, the differentiable loss
reduction, the benchmark reference, the evaluation manifest and the CPU two-way tests. GPU/NPU tests are registered in
`tests/test_shardformer/` and run through the repository's batch scripts.
