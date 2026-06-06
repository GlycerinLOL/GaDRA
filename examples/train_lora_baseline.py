"""LoRA baseline continual pre-training — the paper's baseline, with stock ``peft.LoraConfig``.

This is the **baseline** GaDRA is compared against: a plain LoRA adapter (no gate), trained with the same
data pipeline and recipe as ``examples/train.py`` so the two are directly comparable. It does NOT import
``gadra`` — a LoRA baseline is just standard ``peft`` — and it does NOT modify the GaDRA entry script.

    uv run python -m examples.train_lora_baseline --config examples/config/train_lora.yaml

Same config surface and ``--override`` mechanism as ``examples.train`` (held-out validation, best-checkpoint
selection, W&B tracking, smoke knobs all behave identically); the only differences are the peft config
(``LoraConfig`` instead of ``GaDRAConfig``, so no ``router_conditioning`` / ``gate`` fields) and that LoRA
is mergeable (we do not merge — CPT keeps the adapter attached, matching the paper's eval). The data file
is JSONL with a ``text`` field per line (pre-training format).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys

# Repo-only tooling: support both ``python -m examples.train_lora_baseline`` and the file-path form.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from examples import load_run_config  # noqa: E402

logger = logging.getLogger("gadra.train_lora")


def _is_main_process() -> bool:
    """True on rank 0 (or single-process runs). Reads the launcher's env, no torch.distributed needed."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")) or "0") <= 0


def _setup_logging() -> None:
    """Restore the original ``train_peft.py`` logging surface.

    A timestamped root format plus ``transformers``/``datasets`` verbosity raised to INFO on the main
    process. The verbosity bump is what makes ``Trainer`` emit its ``***** Running training *****``
    summary (num examples, per-device / global batch size, gradient-accumulation steps, total
    optimization steps, trainable params); at the default WARNING level that block is suppressed.
    """
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
    """Resolve ``report_to`` and, when W&B is requested, export project/entity from the config.

    Default is ``none`` so a fresh clone / CI run never blocks on a missing W&B login. Set
    ``report_to: wandb`` (+ ``wandb_project`` / ``wandb_entity``, or the ``WANDB_*`` env vars) to enable.
    """
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
    p = argparse.ArgumentParser(description="LoRA baseline continual pre-training (stock peft; config-driven).")
    p.add_argument("--config", required=True, help="Path to a YAML run-config (see examples/config/train_lora.yaml).")
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
    from peft import LoraConfig, get_peft_model
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

    peft_config = LoraConfig(
        r=int(cfg.get("r", 512)),
        lora_alpha=float(cfg.get("lora_alpha", 1.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("target_modules", ["up_proj", "gate_proj", "down_proj"])),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    if _is_main_process():
        logger.info("model:\n%s", model)  # full module structure (mirrors train.py)
    if gradient_checkpointing:
        model.enable_input_require_grads()  # required for gradient checkpointing with a peft adapter

    max_seq_length = int(cfg.get("max_seq_length", 1024))
    num_proc = int(cfg.get("preprocessing_num_workers", 8))
    raw = load_dataset("json", data_files=cfg["train_file"], split="train")
    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:  # smoke-run subsample (matches the original --max_train_samples)
        raw = raw.select(range(min(int(max_train_samples), len(raw))))
    tokenized = pack_text_dataset(raw, tokenizer, max_length=max_seq_length, strategy=packing, num_proc=num_proc)

    # Optional held-out validation (the original train/valid split): a separate JSONL in the same "text"
    # format, packed identically. When set, it enables eval during training (eval_strategy="steps").
    validation_file = cfg.get("validation_file")
    eval_dataset = None
    if validation_file:
        raw_val = load_dataset("json", data_files=validation_file, split="train")
        eval_dataset = pack_text_dataset(raw_val, tokenizer, max_length=max_seq_length, strategy=packing, num_proc=num_proc)

    if _is_main_process():
        logger.info("train packed examples: %d", len(tokenized))
        logger.info("eval packed examples: %s", len(eval_dataset) if eval_dataset is not None else "none (no validation_file)")
        for i in range(min(3, len(tokenized))):
            logger.info("Sample %d: %s", i, tokenized[i])  # token-id rows (mirrors train.py)

    if packing == "fa2_collator":
        collator = PackingCollator()
        collator.check_model(model)  # hard-assert flash_attention_2 (per-doc varlen needs it)
    else:
        collator = default_data_collator

    # save_strategy="best" (the research-repo default) keeps only the lowest-eval-loss checkpoint, but it
    # needs eval. Degrade to "steps" (with a warning) when no validation set is provided so a fresh run
    # never crashes — provide validation_file to reproduce the original exactly.
    save_strategy = cfg.get("save_strategy", "steps")
    if save_strategy == "best" and eval_dataset is None:
        logger.warning("save_strategy='best' needs a validation_file; falling back to 'steps'.")
        save_strategy = "steps"

    # Gradient-checkpointing reentrancy: null (default) leaves it to the library — matches the original
    # run (resolved gradient_checkpointing_kwargs=None). Set false only if you hit a GC autograd error.
    gc_reentrant = cfg.get("gradient_checkpointing_use_reentrant")
    gc_kwargs = None if gc_reentrant is None else {"use_reentrant": bool(gc_reentrant)}

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=float(cfg.get("num_train_epochs", 3.0)),
        max_steps=int(cfg.get("max_steps", -1)),  # -1 = honor num_train_epochs; >0 caps steps (smoke run)
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
        bf16_full_eval=bool(cfg.get("bf16_full_eval", True)),  # eval in bf16 too (original)
        average_tokens_across_devices=bool(cfg.get("average_tokens_across_devices", True)),  # token-weighted loss in distributed (original)
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs if gradient_checkpointing else None,
        logging_steps=int(cfg.get("logging_steps", 1)),
        save_strategy=save_strategy,
        save_steps=int(cfg.get("save_steps", 100)),
        save_total_limit=int(cfg.get("save_total_limit", 1)),
        metric_for_best_model=cfg.get("metric_for_best_model", "loss"),  # used when save_strategy=best
        # Only the model/adapter is checkpointed — no optimizer / scheduler / RNG / global_step
        # (matches the research repo; keeps 8B + DeepSpeed checkpoints small. Trade-off: can't resume.)
        save_only_model=bool(cfg.get("save_only_model", True)),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(cfg.get("eval_steps", 100)),
        seed=seed,
        # The original used remove_unused_columns=True under SFTTrainer; here the custom PackingCollator
        # consumes the tokenized fields directly, so column pruning must stay off (does not affect numerics).
        remove_unused_columns=False,
        report_to=_configure_tracking(cfg),  # "none" (default) | "wandb" — see train_lora.yaml
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
