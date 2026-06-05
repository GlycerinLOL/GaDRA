"""Data processing for GaDRA continual pre-training — tokenizer build, EOS handling, FA2 packing.

Consolidated (and ported VERBATIM) from the research repo's ``Preprocessing.py`` + ``utils.py`` +
``inference_common.build_tokenizer`` so the packed token stream matches the paper's CPT bit-for-bit
(golden-tested, gate G2). No dependency on the research package (only ``transformers`` / ``torch``).

Three concerns, one module:
  * tokenizer build  — :func:`get_tokenizer_for_preprocess`, :func:`build_tokenizer`, :func:`load_chat_template`
  * EOS handling     — :func:`ensure_eos`, :func:`check_tokenizer_auto_adds_eos`, :func:`get_first_sample_text`
  * packing          — :func:`tokenize_pt`, :func:`group_texts`, :class:`PackingCollator`, :func:`pack_text_dataset`

Quickstart (mirrors ``examples/train.py``)::

    from datasets import load_dataset
    from examples.processing import get_tokenizer_for_preprocess, pack_text_dataset, PackingCollator

    tok = get_tokenizer_for_preprocess("meta-llama/Llama-3.1-8B-Instruct",
                                       chat_template_path="...jinja", truncation_side="right", padding_side="right")
    ds = load_dataset("json", data_files={"train": "corpus.jsonl"})["train"]
    tokenized = pack_text_dataset(ds, tok, max_length=1024, strategy="fa2_collator")
    collator = PackingCollator()        # call collator.check_model(model) once (requires FA2)
"""

from __future__ import annotations

import logging
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------------------
# Tokenizer build (verbatim ``inference_common.build_tokenizer`` + ``utils.get_tokenizer_for_preprocess``)
# --------------------------------------------------------------------------------------------------
def load_chat_template(chat_template_path: Optional[str | Path]) -> Optional[str]:
    """Read a Jinja chat-template file (verbatim ``utils.get_chat_template``)."""
    if chat_template_path is None:
        return None
    path = Path(chat_template_path)
    if not path.exists():
        raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
    return path.read_text(encoding="utf-8")


def build_tokenizer(
    tokenizer_path: str,
    *,
    cache_dir: Optional[str] = None,
    padding_side: Optional[str] = None,
    truncation_side: Optional[str] = None,
    chat_template: Optional[str] = None,
    chat_template_path: Optional[str | Path] = None,
    add_bos_token: Optional[bool] = None,
    add_eos_token: Optional[bool] = None,
    **auto_tokenizer_kwargs: Any,
):
    """Create a tokenizer with shared defaults plus optional chat-template injection.

    Verbatim port of ``inference_common.build_tokenizer`` — pad_token defaults to EOS, sides are applied,
    and a chat template (string or file path) is attached when given.
    """
    pretrained_kwargs = dict(auto_tokenizer_kwargs)
    if cache_dir is not None:
        pretrained_kwargs["cache_dir"] = cache_dir
    if add_bos_token is not None:
        pretrained_kwargs["add_bos_token"] = add_bos_token
    if add_eos_token is not None:
        pretrained_kwargs["add_eos_token"] = add_eos_token

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **pretrained_kwargs)

    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side

    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if chat_template is None and chat_template_path:
        chat_template = load_chat_template(chat_template_path)
    if chat_template:
        tokenizer.chat_template = chat_template

    return tokenizer


def get_tokenizer_for_preprocess(
    model_name_or_path: str,
    chat_template_path: Optional[str | Path] = None,
    truncation_side: Optional[str] = None,
    padding_side: Optional[str] = None,
):
    """Build a tokenizer with the project's preprocessing defaults.

    Verbatim port of ``utils.get_tokenizer_for_preprocess``: ``add_bos_token=True``, ``add_eos_token=True``,
    plus the optional chat template and padding/truncation sides.
    """
    return build_tokenizer(
        tokenizer_path=model_name_or_path,
        padding_side=padding_side,
        truncation_side=truncation_side,
        chat_template_path=str(chat_template_path) if chat_template_path else None,
        add_bos_token=True,
        add_eos_token=True,
    )


# --------------------------------------------------------------------------------------------------
# EOS handling (verbatim ``Preprocessor._ensure_eos`` + ``utils.check_tokenizer_auto_adds_eos`` / ``get_first_sample_text``)
# --------------------------------------------------------------------------------------------------
def ensure_eos(texts, eos_token):
    """Append ``eos_token`` to each sample if it is missing (verbatim ``Preprocessor._ensure_eos``)."""
    if eos_token is None:
        return texts

    def append_eos(text: str) -> str:
        return text if text.endswith(eos_token) else text + eos_token

    if isinstance(texts, list):
        return [append_eos(t) for t in texts]
    return append_eos(texts)


def check_tokenizer_auto_adds_eos(tokenizer, test_text: str = "Hello world!") -> bool:
    """Return ``True`` if the tokenizer appends EOS under ``add_special_tokens=True``.

    Verbatim port of ``utils.check_tokenizer_auto_adds_eos``. Drives the manual-EOS branch in
    :func:`tokenize_pt`: Llama-3.1 returns ``False`` (no auto-EOS) -> EOS appended.
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        logger.warning("No eos_token_id defined.")
        return False
    input_ids = tokenizer(test_text, add_special_tokens=True, padding=False, truncation=False)["input_ids"]
    auto_add_eos = input_ids[-1] == eos_token_id
    logger.info("Auto add eos: %s, eos_token_id: %s", auto_add_eos, eos_token_id)
    return auto_add_eos


def get_first_sample_text(dataset: Sequence[dict[str, Any]]) -> str:
    """Extract the first sample's text to probe EOS behavior (verbatim ``utils.get_first_sample_text``)."""
    if len(dataset) == 0:
        logger.warning("Dataset is empty, using default text.")
        return "Hello world!"

    first_sample = dataset[0]
    if "text" in first_sample:
        text = first_sample["text"]
    elif "content" in first_sample:
        text = first_sample["content"]
    elif "messages" in first_sample:
        if first_sample["messages"]:
            text_parts = [msg["content"] for msg in first_sample["messages"] if "content" in msg]
            text = " ".join(text_parts) if text_parts else "Hello world!"
        else:
            text = "Hello world!"
    else:
        text_fields = [v for v in first_sample.values() if isinstance(v, str)]
        text = text_fields[0] if text_fields else "Hello world!"
    return text


# --------------------------------------------------------------------------------------------------
# PT tokenization + packing (verbatim ``Preprocessor.tokenize_function_pt`` packing branch,
# ``grouping_concate_all``, ``PackingCollator``)
# --------------------------------------------------------------------------------------------------
def tokenize_pt(examples, tokenizer, *, max_length: int, auto_add_eos: bool, column_name: str = "text"):
    """PT/packing tokenize — verbatim ``Preprocessor.tokenize_function_pt`` (packing strategy).

    EOS polarity (the footgun): when the tokenizer does NOT auto-append EOS (``auto_add_eos=False``, e.g.
    Llama-3.1), EOS is appended manually here; when it does, it is left to the tokenizer. Returns a dict
    with ``input_ids`` and ``labels`` (a copy of ``input_ids``).
    """
    texts = examples[column_name]
    eos_token = getattr(tokenizer, "eos_token", None)

    if not auto_add_eos and eos_token is not None:
        # Maintain historical behavior: add EOS even when the tokenizer does not.
        texts = ensure_eos(texts, eos_token)

    inputs = tokenizer(text=texts, add_special_tokens=True, truncation=True, max_length=max_length)
    inputs["labels"] = [list(ids) for ids in inputs["input_ids"]]
    return inputs


def group_texts(examples, block_size: int):
    """Concatenate then chunk into ``block_size`` blocks — verbatim ``Preprocessor.grouping_concate_all``."""
    keys = set(["input_ids", "labels", "attention_mask", "token_ppl"]).intersection(examples.keys())
    concatenated_examples = {k: list(chain(*examples[k])) for k in keys}

    total_length = len(concatenated_examples["input_ids"])
    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    if "labels" not in result.keys():
        result["labels"] = result["input_ids"].copy()
    return result


class PackingCollator:
    """Flatten a batch of variable-length samples into one packed sequence (``batch_size=1``).

    Verbatim port of the research repo's ``Preprocessing.PackingCollator``. ``position_ids`` reset at each
    document boundary so Flash-Attention-2's varlen path (``_prepare_flash_attention_from_position_ids``)
    computes ``cu_seqlens`` and applies per-document causal attention. ``attention_mask`` is OMITTED — FA2
    requires it to be ``None`` to enter the ``position_ids`` varlen branch.

    Safety: under SDPA/eager the missing mask silently degrades to full causal attention *across*
    documents (cross-contamination, wrong numbers, no error). :meth:`check_model` hard-asserts FA2.
    """

    def __init__(self, *, assert_flash_attention: bool = True):
        self.assert_flash_attention = assert_flash_attention

    def check_model(self, model) -> None:
        """Hard-assert the model uses ``flash_attention_2``. Call once before training."""
        impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if self.assert_flash_attention and impl != "flash_attention_2":
            raise ValueError(
                "PackingCollator emits position_ids without an attention_mask, which only yields correct "
                f"per-document causal attention under flash_attention_2; model has "
                f"_attn_implementation={impl!r}. Load the model with attn_implementation='flash_attention_2' "
                "(or construct PackingCollator(assert_flash_attention=False) to override)."
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        packed_ids: List[int] = []
        packed_labels: List[int] = []
        packed_pos: List[int] = []

        for f in features:
            ids = f["input_ids"] if isinstance(f["input_ids"], list) else f["input_ids"].tolist()
            labels = f["labels"] if isinstance(f["labels"], list) else f["labels"].tolist()
            packed_ids.extend(ids)
            packed_labels.extend(labels)
            packed_pos.extend(range(len(ids)))

        return {
            "input_ids": torch.tensor([packed_ids], dtype=torch.long),
            "labels": torch.tensor([packed_labels], dtype=torch.long),
            "position_ids": torch.tensor([packed_pos], dtype=torch.long),
        }


def pack_text_dataset(
    dataset,
    tokenizer,
    *,
    max_length: int,
    strategy: str = "fa2_collator",
    auto_add_eos: Optional[bool] = None,
    column_name: str = "text",
    num_proc: Optional[int] = None,
    block_size: Optional[int] = None,
):
    """Tokenize a text dataset (a single ``datasets.Dataset``) for GaDRA CPT.

    * ``strategy='fa2_collator'`` (default): per-sample truncate to ``max_length``; pair the returned
      dataset with :class:`PackingCollator` at train time.
    * ``strategy='group'``: additionally concatenate + chunk into ``block_size`` (default ``max_length``)
      blocks (standard run_clm); use a plain default collator at train time.

    ``auto_add_eos``: if ``None``, auto-detected via :func:`check_tokenizer_auto_adds_eos` on the first
    sample (mirrors the paper's ``train_gadra_cpt.py``).
    """
    if auto_add_eos is None:
        auto_add_eos = check_tokenizer_auto_adds_eos(tokenizer, get_first_sample_text(dataset))

    tokenized = dataset.map(
        tokenize_pt,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": max_length,
            "auto_add_eos": auto_add_eos,
            "column_name": column_name,
        },
        desc="examples.processing tokenize (pt/packing)",
    )

    if strategy == "fa2_collator":
        return tokenized
    if strategy == "group":
        bs = block_size or max_length
        return tokenized.map(
            group_texts,
            batched=True,
            num_proc=num_proc,
            fn_kwargs={"block_size": bs},
            desc="examples.processing group_texts",
        )
    raise ValueError(f"Unknown strategy {strategy!r}; expected 'fa2_collator' or 'group'.")
