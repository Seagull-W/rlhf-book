# Preference RM run log — 2026-09-08

## Command

```bash
uv run python -m reward_models.train_preference_rm \
    --config reward_models/configs/preference_rm.yaml
```

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB total; about 1,419 MiB in use before the run).
- Config: `Qwen/Qwen3-0.6B-Base`, full-backbone training (`freeze_backbone: false`), `batch_size: 2`, `grad_accum_steps: 8`, `max_length: 512`, 5,000 requested preference pairs, seed 123.
- W&B was disabled because `WANDB_PROJECT` was not set.

## Observed progress

- Dataset preparation completed. Of 5,000 selected UltraFeedback pairs, 352 had identical chosen/rejected token IDs after 512-token truncation and were correctly discarded.
- The resulting split was 4,183 training pairs and 465 validation pairs.
- Model loading completed with 596.05M trainable parameters. The configured two epochs correspond to 524 optimizer steps, with 52 warm-up steps.
- The first optimizer step completed: `Epoch 0 step 1 | loss 0.5816 | acc 0.750`.

## Error

The run exited with status 1 during a subsequent `loss.backward()` call:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 594.00 MiB.
GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free.
```

The allocator emitted earlier failed allocations of approximately 594 MiB and 20 MiB. GPU utilization was 100% and memory use was 7,836 MiB immediately before failure.

## Cause

This configuration performs full FP32 fine-tuning of the 0.6B backbone. The model parameters, gradients, AdamW optimizer state, and training activations exceed the available 8 GiB of VRAM. Lowering only the sample count will not fix this, since the allocation that failed happens within one training step.

## Resolution options

1. To preserve full-backbone training, run this unchanged configuration on a GPU with at least 16 GiB VRAM; 20–25 GiB provides more headroom for the 512-token sequences and batch size 2.
2. To run on this 8 GiB GPU, copy the configuration to a local small-run YAML and set `freeze_backbone: true`. This makes the run head-only training, so it is a functional demonstration but is not directly comparable with the documented full-finetuning reference. Set `batch_size: 1` and reduce `max_length` if activation memory still approaches the limit.

Example 8 GiB-oriented config changes:

```yaml
freeze_backbone: true
batch_size: 1
max_length: 384
samples: 500
epochs: 1
use_wandb: false
skip_demo: true
```

Use it with:

```bash
uv run python -m reward_models.train_preference_rm \
    --config reward_models/configs/preference_rm_8gb_demo.yaml
```

The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocator hint shown in the exception may mitigate fragmentation, but it cannot make this full-finetuning configuration fit in 8 GiB and should not be treated as the primary fix.
