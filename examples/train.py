"""Continual pre-training — config-driven entry for every adapter method (standard ``transformers.Trainer``).

The adapter is chosen by the run-config's ``method:`` field (gadra | gadra-mono | gadra-soft | lora),
dispatched by ``examples/methods.py``. GaDRA trains with the plain causal-LM objective (no custom trainer,
no auxiliary loss). Configuration is read from a YAML run-config (``examples/config/train.yaml``); the
training file is JSONL with a ``text`` field per line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

# Support both ``python -m examples.train`` and ``python examples/train.py`` by making the repo root importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from examples import load_run_config  # noqa: E402
from examples.methods import build_peft_config  # noqa: E402

logger = logging.getLogger("gadra.train")


def _is_main_process() -> bool:
    """True on rank 0 (or single-process runs); reads the launcher's env."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) or "0") <= 0


def _setup_logging() -> None:
    """Set up a timestamped root format and raise ``transformers``/``datasets`` verbosity to INFO on rank 0."""
    import datasets
    import transformers

    level = logging.INFO if _is_main_process() else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=level,
    )
    logger.setLevel(level)
    datasets.utils.logging.set_verbosity(level)
    transformers.utils.logging.set_verbosity(level)
    if _is_main_process():
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()


def _configure_tracking(cfg: dict) -> str | list:
    """Resolve ``report_to`` (default ``none``) and, when W&B is requested, export project/entity from the config."""
    report_to = cfg.get("report_to", "none")
    uses_wandb = "wandb" in (report_to if isinstance(report_to, (list, tuple)) else [report_to])
    if uses_wandb:
        project = cfg.get("wandb_project") or os.environ.get("WANDB_PROJECT")
        entity = cfg.get("wandb_entity") or os.environ.get("WANDB_ENTITY")
        if project:
            os.environ["WANDB_PROJECT"] = str(project)
        if entity:
            os.environ["WANDB_ENTITY"] = str(entity)
    return report_to


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continual pre-training, config-driven (adapter chosen by the config's method: field).")
    p.add_argument("--config", required=True, help="Path to a YAML run-config (see examples/config/train.yaml).")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config field (YAML-typed value); repeatable.")
    return p.parse_args()


def _load_model(model_name_or_path: str, packing: str):
    """Load the base model with the attention implementation the packing strategy requires."""
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

    _setup_logging()
    logger.info("Run config:\n%s", json.dumps(cfg, indent=2, default=str, sort_keys=True))

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

    peft_config = build_peft_config(cfg)  # method: gadra | gadra-mono | gadra-soft | lora
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if _is_main_process():
        logger.info("model:\n%s", model)
    if gradient_checkpointing:
        model.enable_input_require_grads()  # required for gradient checkpointing with a peft adapter

    max_seq_length = int(cfg.get("max_seq_length", 1024))
    num_proc = int(cfg.get("preprocessing_num_workers", 8))
    raw = load_dataset("json", data_files=cfg["train_file"], split="train")
    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:  # smoke-run subsample
        raw = raw.select(range(min(int(max_train_samples), len(raw))))
    tokenized = pack_text_dataset(raw, tokenizer, max_length=max_seq_length, strategy=packing, num_proc=num_proc)

    # Optional held-out validation: a separate JSONL in the same "text" format, packed identically.
    # When set, it enables eval during training (eval_strategy="steps").
    validation_file = cfg.get("validation_file")
    eval_dataset = None
    if validation_file:
        raw_val = load_dataset("json", data_files=validation_file, split="train")
        eval_dataset = pack_text_dataset(raw_val, tokenizer, max_length=max_seq_length, strategy=packing, num_proc=num_proc)

    if _is_main_process():
        logger.info("train packed examples: %d", len(tokenized))
        logger.info("eval packed examples: %s", len(eval_dataset) if eval_dataset is not None else "none (no validation_file)")
        for i in range(min(3, len(tokenized))):
            logger.info("Sample %d: %s", i, tokenized[i])

    if packing == "fa2_collator":
        collator = PackingCollator()
        collator.check_model(model)  # hard-assert flash_attention_2 (per-doc varlen needs it)
    else:
        collator = default_data_collator

    # save_strategy="best" keeps only the lowest-eval-loss checkpoint but needs eval; degrade to "steps"
    # (with a warning) when no validation set is provided.
    save_strategy = cfg.get("save_strategy", "steps")
    if save_strategy == "best" and eval_dataset is None:
        logger.warning("save_strategy='best' needs a validation_file; falling back to 'steps'.")
        save_strategy = "steps"

    # Gradient-checkpointing reentrancy: null (default) leaves it to the library.
    gc_reentrant = cfg.get("gradient_checkpointing_use_reentrant")
    gc_kwargs = None if gc_reentrant is None else {"use_reentrant": bool(gc_reentrant)}

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=float(cfg.get("num_train_epochs", 3.0)),
        max_steps=int(cfg.get("max_steps", -1)),  # -1 = honor num_train_epochs; >0 caps steps
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", cfg.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        max_grad_norm=float(cfg.get("max_grad_norm", 0.1)),
        optim=cfg.get("optim", "adamw_torch"),
        bf16=True,
        bf16_full_eval=bool(cfg.get("bf16_full_eval", True)),  # eval in bf16 too
        average_tokens_across_devices=bool(cfg.get("average_tokens_across_devices", True)),  # token-weighted loss in distributed
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs if gradient_checkpointing else None,
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_strategy=save_strategy,
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 1)),
        metric_for_best_model=cfg.get("metric_for_best_model", "loss"),  # used when save_strategy=best
        # Only the model/adapter is checkpointed — no optimizer / scheduler / RNG / global_step (can't resume).
        save_only_model=bool(cfg.get("save_only_model", True)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 100)),
        seed=seed,
        # PackingCollator consumes the tokenized fields directly, so column pruning must stay off.
        remove_unused_columns=False,
        report_to=_configure_tracking(cfg),  # "none" (default) | "wandb"
        run_name=cfg.get("run_name") or os.path.basename(cfg["output_dir"].rstrip("/")),
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized, eval_dataset=eval_dataset, data_collator=collator
    )
    trainer.train()
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])


if __name__ == "__main__":
    main()
