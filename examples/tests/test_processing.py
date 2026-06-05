"""G2 — data-path parity: ``examples.processing`` reproduces the research repo's packed token stream bit-for-bit.

The golden (``examples/tests/_golden/data_golden.pt``) is captured by ``capture_data_golden.py`` from the parent
``Preprocessing`` on real BBC pretraining samples. Tests (a) packing and (b) grouping are hermetic (pure
python/torch, no tokenizer); test (c) re-tokenizes via ``examples.processing`` and is gated on the frozen tokenizer
dir being present (the maintainer/dev host), skipping cleanly on a fresh clone.
"""

import os
import pathlib

import pytest
import torch

from examples.processing import (
    PackingCollator,
    check_tokenizer_auto_adds_eos,
    get_first_sample_text,
    get_tokenizer_for_preprocess,
    group_texts,
    tokenize_pt,
)

GOLDEN = pathlib.Path(__file__).parent / "_golden" / "data_golden.pt"
pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("data_golden.pt missing — run capture_data_golden.py with GADRA_SOURCE_REPO set")
    return torch.load(GOLDEN, weights_only=False)


def test_packing_collator_matches_golden(golden):
    """The FA2 varlen packing layout (batch_size=1, per-doc position_ids, no mask) is bit-exact."""
    out = PackingCollator(assert_flash_attention=False)(golden["sample_features"])
    assert out["input_ids"].tolist() == golden["packed"]["input_ids"]
    assert out["labels"].tolist() == golden["packed"]["labels"]
    assert out["position_ids"].tolist() == golden["packed"]["position_ids"]
    # structural invariants of the FA2 packing layout
    assert out["input_ids"].shape[0] == 1, "PackingCollator must emit batch_size == 1"
    assert "attention_mask" not in out, "FA2 varlen requires no attention_mask"
    assert out["position_ids"].shape == out["input_ids"].shape


def test_group_texts_matches_golden(golden):
    """The generic group_texts strategy reproduces grouping_concate_all."""
    feats = {
        "input_ids": [f["input_ids"] for f in golden["sample_features"]],
        "labels": [f["labels"] for f in golden["sample_features"]],
    }
    out = group_texts(feats, block_size=golden["meta"]["max_len"])
    assert out["input_ids"] == golden["grouped"]["input_ids"]
    assert out["labels"] == golden["grouped"]["labels"]


def test_tokenize_pt_and_eos_match_golden(golden):
    """tokenize_pt + EOS-polarity detection reproduce the parent per-sample token ids."""
    tok_dir = golden["meta"]["tokenizer_dir"]
    if not os.path.isdir(tok_dir):
        pytest.skip(f"frozen tokenizer dir absent ({tok_dir}); hermetic packing/grouping already covered")
    tokenizer = get_tokenizer_for_preprocess(tok_dir, truncation_side="right", padding_side="right")
    # EOS-bool polarity must match the captured value (Llama-3.1 -> False -> EOS appended manually)
    auto = check_tokenizer_auto_adds_eos(tokenizer, get_first_sample_text([{"text": t} for t in golden["texts"]]))
    assert bool(auto) == golden["meta"]["auto_add_eos"]
    out = tokenize_pt(
        {"text": golden["texts"]},
        tokenizer,
        max_length=golden["meta"]["max_len"],
        auto_add_eos=auto,
    )
    for i, feat in enumerate(golden["sample_features"]):
        assert list(out["input_ids"][i]) == feat["input_ids"], f"input_ids mismatch at sample {i}"
        assert list(out["labels"][i]) == feat["labels"], f"labels mismatch at sample {i}"
