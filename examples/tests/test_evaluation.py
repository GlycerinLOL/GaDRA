"""G3 — scorer parity: ``examples.evaluation`` reproduces the research repo's deterministic scorers.

The golden (``examples/tests/_golden/eval_golden.pt``) is captured by ``capture_eval_golden.py`` from the parent
scorers on a diverse case set (exact/partial/empty QA, currency/comma/negative/float GSM8K, fenced/special-
token MBPP code). Hermetic — no model, tokenizer, or source repo needed at test time.
"""

import pathlib

import pytest
import torch

from examples.evaluation import (
    compute_gsm8k_metrics,
    compute_qa_metrics,
    llama_gsm8k_parse,
    llama_mbpp_parse,
    normalize_answer,
    normalize_numeric_answer,
)

GOLDEN = pathlib.Path(__file__).parent / "_golden" / "eval_golden.pt"
pytestmark = pytest.mark.parity


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip("eval_golden.pt missing — run capture_eval_golden.py with GADRA_SOURCE_REPO set")
    return torch.load(GOLDEN, weights_only=False)


def test_qa_metrics_match_golden(golden):
    for c in golden["qa"]:
        out = compute_qa_metrics(c["pred"], c["ref"])
        assert out["f1"] == c["f1"], f"f1 mismatch for {c['pred']!r} / {c['ref']!r}"
        assert out["em"] == c["em"], f"em mismatch for {c['pred']!r} / {c['ref']!r}"


def test_gsm8k_metrics_match_golden(golden):
    for c in golden["gsm"]:
        out = compute_gsm8k_metrics(c["pred"], c["answers"])
        assert out == c["result"], f"gsm8k metric mismatch for {c['pred']!r}"


def test_parsers_match_golden(golden):
    for c in golden["mbpp_parse"]:
        assert llama_mbpp_parse(c["raw"]) == c["out"], f"mbpp parse mismatch for {c['raw']!r}"
    for c in golden["gsm_parse"]:
        assert llama_gsm8k_parse(c["raw"]) == c["out"], f"gsm parse mismatch for {c['raw']!r}"


def test_normalizers_match_golden(golden):
    for c in golden["norm"]:
        assert normalize_answer(c["s"]) == c["answer"], f"normalize_answer mismatch for {c['s']!r}"
        assert normalize_numeric_answer(c["s"]) == c["numeric"], f"normalize_numeric mismatch for {c['s']!r}"


# Source-independent invariants (do not rely on the golden being present).
def test_qa_exact_match_is_100():
    out = compute_qa_metrics("Paris", "paris")
    assert out["em"] == 100.0 and out["f1"] == 100.0


def test_gsm8k_currency_comma_normalization():
    out = compute_gsm8k_metrics("The final answer is $1,234", ["1234"])
    assert out["is_correct"] and out["em"] == 100.0
