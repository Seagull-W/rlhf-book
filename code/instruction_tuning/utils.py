# Utilities for Instruction Tuning (SFT).

import os
import platform
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from .config import Config


console = Console()

IGNORE_INDEX = -100

DEFAULT_SAMPLE_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "How does photosynthesis work?",
]

TOKEN_COLORS = {
    "<|endoftext|>": "bold red",
    "<|pad|>": "bold magenta",
    "<|user|>": "bold blue",
    "<|assistant|>": "bold green",
    "<|system|>": "bold yellow",
}


def _colorize_tokens(text: str) -> str:
    text = escape(text)
    for marker, style in TOKEN_COLORS.items():
        text = text.replace(marker, f"[{style}]{marker}[/{style}]")
    return text


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_attn_implementation() -> str:
    if platform.machine() != "x86_64":
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def resolve_device(cuda_device_id: int = 0, device: str = "auto") -> torch.device:
    """Resolve 'auto' to CUDA if available, otherwise CPU."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available, but device is set to 'cuda'")
    elif device != "cpu":
        raise ValueError(f"Unsupported device={device!r}. Expected 'auto', 'cuda', or 'cpu'.")

    if device == "cuda":
        device = f"cuda:{cuda_device_id}"
    return torch.device(device)


def load_model(cfg: Config, device: torch.device):
    attn_impl = get_attn_implementation()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=False)
    if tokenizer.chat_template is None and cfg.chat_template_source:
        donor = AutoTokenizer.from_pretrained(cfg.chat_template_source, trust_remote_code=False)
        if donor.chat_template is None:
            raise ValueError(
                f"chat_template_source {cfg.chat_template_source} has no chat_template."
            )
        tokenizer.chat_template = donor.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=False,
        attn_implementation=attn_impl,
        torch_dtype=dtype,
    ).to(device)
    if cfg.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.lora_target_modules,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if cfg.use_lora:
            model.enable_input_require_grads()
    return model, tokenizer


@dataclass
class SFTBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device | str) -> "SFTBatch":
        return SFTBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
        )


def compute_loss(
    model,
    batch: "SFTBatch",
    *,
    end_token_id: int | None = None,
    end_token_weight: float = 1.0,
) -> torch.Tensor:
    """Causal-LM CE with prompt masking and optional end-of-turn emphasis."""
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = batch.labels[:, 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    token_losses = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    valid = flat_labels != IGNORE_INDEX
    weights = torch.ones_like(token_losses)
    if end_token_id is not None and end_token_weight != 1.0:
        weights = torch.where(
            flat_labels == end_token_id,
            torch.full_like(weights, end_token_weight),
            weights,
        )
    return (token_losses[valid] * weights[valid]).sum() / weights[valid].sum()


def _encode_batch(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    drop_overlong: bool,
) -> dict[str, list[list[int]]]:
    """Render conversations and mask all but the final assistant turn."""
    conversations = [
        messages
        for messages in batch["messages"]
        if messages and messages[-1]["role"] == "assistant"
    ]
    if not conversations:
        return {"input_ids": [], "labels": []}

    template_kwargs = {
        "tokenize": True,
        "return_dict": False,
        "truncation": not drop_overlong,
    }
    if not drop_overlong:
        template_kwargs["max_length"] = max_length
    prompt_ids_batch = tokenizer.apply_chat_template(
        [messages[:-1] for messages in conversations],
        add_generation_prompt=True,
        **template_kwargs,
    )
    full_ids_batch = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=False,
        **template_kwargs,
    )

    input_ids, labels = [], []
    for prompt_ids, full_ids in zip(prompt_ids_batch, full_ids_batch, strict=True):
        if drop_overlong and len(full_ids) > max_length:
            continue
        prompt_length = min(len(prompt_ids), len(full_ids))
        if prompt_length == len(full_ids):
            continue
        input_ids.append(full_ids)
        labels.append([IGNORE_INDEX] * prompt_length + full_ids[prompt_length:])
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, encoded: list[dict[str, torch.Tensor]]):
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.encoded[idx]


def _collate(examples: list[dict[str, torch.Tensor]], pad_token_id: int) -> SFTBatch:
    max_len = max(ex["input_ids"].size(0) for ex in examples)
    input_ids, attention_mask, labels = [], [], []
    for ex in examples:
        ids, lbl = ex["input_ids"], ex["labels"]
        pad = max_len - ids.size(0)
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(
            torch.cat(
                [torch.ones(ids.size(0), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]
            )
        )
        labels.append(torch.cat([lbl, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
    return SFTBatch(
        input_ids=torch.stack(input_ids),
        attention_mask=torch.stack(attention_mask),
        labels=torch.stack(labels),
    )


def create_dataloader(
    cfg: Config,
    tokenizer: PreTrainedTokenizer,
    *,
    split: str | None = None,
    max_samples: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    split = split or cfg.dataset_split
    max_samples = cfg.max_samples if max_samples is None else max_samples
    raw = load_dataset(cfg.dataset_name, split=split)
    if cfg.shuffle_before_select:
        raw = raw.shuffle(seed=cfg.seed)
    if max_samples is not None and len(raw) > max_samples:
        raw = raw.select(range(max_samples))

    encoded = raw.map(
        _encode_batch,
        batched=True,
        batch_size=256,
        remove_columns=raw.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": cfg.max_length,
            "drop_overlong": cfg.drop_overlong,
        },
        load_from_cache_file=True,
        desc=f"Encoding {split} conversations",
    )
    skipped = len(raw) - len(encoded)
    encoded = encoded.with_format("torch")
    if not encoded:
        raise RuntimeError(f"No trainable rows after tokenization for split={split}.")
    if skipped:
        console.print(f"[dim]Skipped {skipped}/{len(raw)} rows from {split}.[/dim]")
    return DataLoader(
        SFTDataset(encoded),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: _collate(batch, tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=False,
    )


def generate_samples(
    model,
    tokenizer: PreTrainedTokenizer,
    cfg: Config,
    step: int,
    prompts: list[str] | None = None,
    max_new_tokens: int | None = None,
) -> list[dict]:
    was_training = model.training
    model.eval()
    new_tokens = max_new_tokens if max_new_tokens is not None else cfg.sample_max_tokens
    prompts = prompts or DEFAULT_SAMPLE_PROMPTS
    console.rule(f"[bold yellow]Samples @ step {step}[/bold yellow]", style="yellow")

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = {tokenizer.eos_token_id}
    if isinstance(im_end_id, int) and im_end_id >= 0:
        eos_ids.add(im_end_id)
    eos_ids.discard(None)

    records = []
    for prompt_id, prompt in enumerate(prompts, start=1):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.sample_max_input_tokens,
        ).to(model.device)
        kwargs = dict(
            **inputs,
            max_new_tokens=new_tokens,
            do_sample=cfg.sample_do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=sorted(eos_ids),
        )
        if cfg.sample_do_sample:
            kwargs.update(temperature=cfg.sample_temperature, top_p=cfg.sample_top_p)
        with torch.no_grad():
            out = model.generate(**kwargs)

        prompt_tokens = inputs.input_ids.shape[1]
        completion_ids = out[0, prompt_tokens:]
        full_text = tokenizer.decode(out[0], skip_special_tokens=False)
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        final_token_id = int(completion_ids[-1]) if completion_ids.numel() else None
        records.append(
            {
                "event": "sample",
                "step": step,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "completion": completion_text,
                "full_text": full_text,
                "new_tokens": int(completion_ids.numel()),
                "terminated": final_token_id in eos_ids,
                "final_token_id": final_token_id,
            }
        )
        console.print(
            Panel(
                _colorize_tokens(full_text),
                title=f"[bold cyan]Prompt {prompt_id}[/bold cyan]",
                title_align="left",
                border_style="cyan",
            )
        )
    if was_training:
        model.train()
    return records


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader) -> float:
    """Return token-weighted loss over a fixed validation dataloader."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in dataloader:
        batch = batch.to(model.device)
        tokens = int((batch.labels[:, 1:] != IGNORE_INDEX).sum().item())
        if tokens:
            total_loss += compute_loss(model, batch).item() * tokens
            total_tokens += tokens
    if was_training:
        model.train()
    return total_loss / max(1, total_tokens)


def progress_bar() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("*"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


def print_training_info(model, cfg: Config, total_steps: int, warmup_steps: int) -> None:
    console.print(
        Panel(
            f"[bold magenta]Model:[/bold magenta] {cfg.model_name}\n"
            f"[dim]Parameters:[/dim] {sum(p.numel() for p in model.parameters()):,}\n"
            f"[dim]Device:[/dim] {model.device}\n"
            f"[dim]Dataset:[/dim] {cfg.dataset_name} (split={cfg.dataset_split})\n"
            f"[dim]Effective batch:[/dim] {cfg.batch_size} x {cfg.gradient_accumulation_steps}"
            f" = {cfg.batch_size * cfg.gradient_accumulation_steps}\n"
            f"[dim]Steps:[/dim] {total_steps} total, {warmup_steps} warmup",
            title="[bold magenta]SFT Configuration[/bold magenta]",
            border_style="magenta",
        )
    )
    console.print(
        "[dim yellow]Note: OLMo-2 reuses [bold]<|endoftext|>[/bold] as BOS, EOS,"
        " and UNK, so it appears at the start *and* end of every conversation.[/dim yellow]"
    )


def print_epoch_header(epoch_idx: int, total_epochs: int) -> None:
    console.rule(f"[bold cyan]Epoch {epoch_idx + 1}/{total_epochs}[/bold cyan]", style="cyan")


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps + 1))
        remaining = total_steps - step
        return max(0.0, remaining / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
