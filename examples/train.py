"""Continual pre-training with GaDRA — single config-driven entry (standard ``transformers.Trainer``).

GaDRA needs no custom trainer and no auxiliary loss — it trains with the plain causal-LM objective. This
reads a YAML run-config (``examples/config/train.yaml``) so the launch surface is the config, not a flag
soup. The recipe mirrors the paper (§A.1): AdamW, cosine schedule, BF16, sequence length 1024, gradient
clipping 0.1, weight decay 0.01, seed 42, 3 epochs.

Single-GPU::

    uv run python -m examples.train --config examples/config/train.yaml

Multi-GPU (= the paper's workflow — ``accelerate launch`` + DeepSpeed ZeRO-2, processes = GPUs)::

    uv run accelerate launch --num_processes <N> \
        --config_file examples/config/deepspeed_zero2.yaml \
        -m examples.train --config examples/config/train.yaml

Ad-hoc overrides without editing the config: ``--override learning_rate=5e-4 --override packing=group``.

Data packing (``packing:`` in the config):
  * ``fa2_collator`` (default, paper-faithful) — per-document varlen packing; REQUIRES flash-attention-2.
  * ``group`` (portable) — concatenate-and-chunk under SDPA; no flash-attn (not bit-exact to the paper).

The training file is JSONL with a ``text`` field per line (pre-training format).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Repo-only tooling: support both ``python -m examples.train`` and ``python examples/train.py``
# (the latter is what some SLURM file-path wrappers use) by ensuring the repo root is importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gadra  # noqa: E402, F401  (registers the "gadra" method with peft)
from examples import load_run_config  # noqa: E402
from gadra import GaDRAConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continual pre-training with GaDRA (config-driven).")
    p.add_argument("--config", required=True, help="Path to a YAML run-config (see examples/config/train.yaml).")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config field (YAML-typed value); repeatable.")
    return p.parse_args()


def _load_model(model_name_or_path: str, packing: str):
    """Load the base model with the attention impl the packing strategy requires (clear error if FA2 absent)."""
    import torch
    from transformers import AutoModelForCausalLM

    attn_impl = "flash_attention_2" if packing == "fa2_collator" else "sdpa"
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation=attn_impl, use_cache=False
        )
    except (ImportError, ValueError) as exc:
        if attn_impl != "flash_attention_2":
            raise
        raise RuntimeError(
            "packing='fa2_collator' (the paper-faithful default) needs flash-attention-2. Install the GPU "
            "deps with `uv sync --group gpu` (they include the prebuilt flash-attn wheel), OR use the "
            "portable path `--override packing=group` (SDPA, no flash-attn; NOT bit-exact to the paper).\n"
            f"underlying error: {exc}"
        ) from exc


def main() -> None:
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, default_data_collator, set_seed

    from examples.processing import PackingCollator, get_tokenizer_for_preprocess, pack_text_dataset

    args = parse_args()
    cfg = load_run_config(args.config, args.override)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    packing = cfg.get("packing", "fa2_collator")
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", True))

    tokenizer = get_tokenizer_for_preprocess(
        cfg["model_name_or_path"],
        chat_template_path=cfg.get("chat_template_path"),
        truncation_side="right",
        padding_side="right",
    )

    model = _load_model(cfg["model_name_or_path"], packing)

    peft_config = GaDRAConfig(
        r=int(cfg.get("r", 512)),
        lora_alpha=float(cfg.get("lora_alpha", 1.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("target_modules", ["up_proj", "gate_proj", "down_proj"])),
        router_conditioning=cfg.get("router_conditioning", "dual"),
        gate=cfg.get("gate", "hard"),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if gradient_checkpointing:
        model.enable_input_require_grads()  # required for gradient checkpointing with a peft adapter

    raw = load_dataset("json", data_files=cfg["train_file"], split="train")
    max_seq_length = int(cfg.get("max_seq_length", 1024))
    tokenized = pack_text_dataset(raw, tokenizer, max_length=max_seq_length, strategy=packing)

    if packing == "fa2_collator":
        collator = PackingCollator()
        collator.check_model(model)  # hard-assert flash_attention_2 (per-doc varlen needs it)
    else:
        collator = default_data_collator

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=float(cfg.get("num_train_epochs", 3.0)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 16)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type="cosine",
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.1)),
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=int(cfg.get("save_steps", 100)),
        seed=seed,
        remove_unused_columns=False,  # the collator consumes the tokenized fields directly
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
