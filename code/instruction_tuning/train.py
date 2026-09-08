import argparse
import json
import os
import time
from pathlib import Path

import torch
import wandb
from torch.nn.utils import clip_grad_norm_

from .config import Config, load_config
from .utils import (
    compute_loss,
    create_dataloader,
    evaluate_loss,
    generate_samples,
    load_model,
    make_lr_scheduler,
    print_epoch_header,
    print_training_info,
    progress_bar,
    resolve_device,
    seed_everything,
)


def main(cfg: Config):
    seed_everything(cfg.seed)
    device = resolve_device(cfg.model_device_id, cfg.device)
    model, tokenizer = load_model(cfg, device)
    train_loader = create_dataloader(cfg, tokenizer)
    validation_loader = None
    if cfg.validation_split and cfg.max_validation_samples:
        validation_loader = create_dataloader(
            cfg,
            tokenizer,
            split=cfg.validation_split,
            max_samples=cfg.max_validation_samples,
            shuffle=False,
        )

    end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    accum = cfg.gradient_accumulation_steps
    steps_per_epoch = (len(train_loader) + accum - 1) // accum
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = make_lr_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    wandb_project = os.environ.get("WANDB_PROJECT", cfg.wandb_project)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", cfg.wandb_run_name)
    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project, name=wandb_run_name, config=vars(cfg))

    metrics_log = None
    if cfg.metrics_log_path:
        metrics_path = Path(cfg.metrics_log_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_log = metrics_path.open("w", encoding="utf-8")
    samples_log = None
    if cfg.samples_log_path:
        samples_path = Path(cfg.samples_log_path)
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        samples_log = samples_path.open("w", encoding="utf-8")

    def write_metric(record: dict) -> None:
        if metrics_log:
            metrics_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics_log.flush()

    def record_samples(step: int) -> dict:
        records = generate_samples(model, tokenizer, cfg, step=step)
        if samples_log:
            for record in records:
                samples_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            samples_log.flush()
        summary = {
            "sample/termination_rate": sum(r["terminated"] for r in records) / len(records),
            "sample/avg_new_tokens": sum(r["new_tokens"] for r in records) / len(records),
        }
        wandb.log(summary, step=step)
        write_metric({"event": "sample_eval", "step": step, **summary})
        return summary

    def record_validation(step: int) -> float | None:
        if validation_loader is None:
            return None
        loss = evaluate_loss(model, validation_loader)
        wandb.log({"validation/loss": loss}, step=step)
        write_metric({"event": "validation", "step": step, "validation/loss": loss})
        return loss

    trainable_dtypes = sorted({str(p.dtype) for p in trainable_params})
    write_metric(
        {
            "event": "run_start",
            "config": vars(cfg),
            "trainable_parameters": sum(p.numel() for p in trainable_params),
            "trainable_dtypes": trainable_dtypes,
        }
    )
    print_training_info(model, cfg, total_steps, warmup_steps)
    console_message = (
        f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}; "
        f"dtypes: {', '.join(trainable_dtypes)}"
    )
    print(console_message, flush=True)

    start_time = time.time()
    global_step = 0
    total_train_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    record_samples(step=0)
    record_validation(step=0)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_nll = 0.0
    accumulated_tokens = 0

    for epoch in range(cfg.num_epochs):
        print_epoch_header(epoch, cfg.num_epochs)
        with progress_bar() as progress:
            task = progress.add_task("Training", total=len(train_loader))
            for batch_idx, batch in enumerate(train_loader):
                batch = batch.to(device)
                loss = compute_loss(
                    model,
                    batch,
                    end_token_id=end_token_id,
                    end_token_weight=cfg.end_token_weight,
                )
                chunk_start = (batch_idx // accum) * accum
                chunk_size = min(accum, len(train_loader) - chunk_start)
                tokens = int((batch.labels[:, 1:] != -100).sum().item())
                if loss.isfinite():
                    (loss / chunk_size).backward()
                    accumulated_nll += loss.item() * tokens
                    accumulated_tokens += tokens
                    total_train_tokens += tokens

                should_step = (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader)
                if should_step:
                    grad_norm = clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    elapsed = time.time() - start_time
                    avg_loss = accumulated_nll / max(1, accumulated_tokens)
                    metrics = {
                        "event": "train_step",
                        "step": global_step,
                        "loss": avg_loss,
                        "grad_norm": float(grad_norm),
                        "learning_rate": scheduler.get_last_lr()[0],
                        "epoch": epoch + (batch_idx + 1) / len(train_loader),
                        "hours": elapsed / 3600,
                        "train/tokens": total_train_tokens,
                        "train/tokens_per_second": total_train_tokens / max(elapsed, 1e-6),
                        "gpu/max_allocated_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                        "gpu/max_reserved_gib": (
                            torch.cuda.max_memory_reserved(device) / 2**30
                            if device.type == "cuda"
                            else 0.0
                        ),
                    }
                    wandb.log({k: v for k, v in metrics.items() if k not in {"event", "step"}}, step=global_step)
                    write_metric(metrics)
                    progress.update(
                        task,
                        advance=1,
                        description=f"[dim]Loss: {avg_loss:.4f}[/dim]",
                    )
                    accumulated_nll = 0.0
                    accumulated_tokens = 0

                    if cfg.sample_every > 0 and global_step % cfg.sample_every == 0:
                        record_samples(global_step)
                    if cfg.eval_every > 0 and global_step % cfg.eval_every == 0:
                        record_validation(global_step)
                    model.train()
                else:
                    progress.update(task, advance=1)

    if cfg.sample_every <= 0 or global_step % cfg.sample_every:
        record_samples(global_step)
    final_validation_loss = None
    if validation_loader is not None:
        final_validation_loss = record_validation(global_step)

    if cfg.output_dir:
        output_path = Path(cfg.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

    write_metric(
        {
            "event": "run_end",
            "steps": global_step,
            "final_validation_loss": final_validation_loss,
            "output_dir": cfg.output_dir,
        }
    )
    if metrics_log:
        metrics_log.close()
    if samples_log:
        samples_log.close()
    wandb.finish()


def main_cli():
    parser = argparse.ArgumentParser(description="Instruction-tune a base model (SFT).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.device = args.device
    main(cfg)


if __name__ == "__main__":
    main_cli()
