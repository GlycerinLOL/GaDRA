"""Capture the G3 scorer golden from the research repo's verbatim scorers (maintainer-only).

Freezes the parent scorers' outputs on a diverse case set so ``test_evaluation.py`` can assert
``examples.evaluation`` reproduces them exactly. Run once where the original repo is importable::

    GADRA_SOURCE_REPO=/path/to/Adapter_Research python GaDRA/examples/tests/capture_eval_golden.py

Writes ``examples/tests/_golden/eval_golden.pt`` (committed). Skips cleanly when the source repo is absent.
"""

import os
import pathlib
import sys

import torch

REPO = os.environ.get("GADRA_SOURCE_REPO")
if not REPO or not pathlib.Path(REPO).is_dir():
    sys.exit("capture_eval_golden: set GADRA_SOURCE_REPO to the research-repo root (maintainer-only).")
sys.path.insert(0, REPO)

from evaluate_document_qa import compute_metrics  # noqa: E402  parent char-F1/EM
from evaluate_gsm8k import compute_gsm8k_metrics, normalize_numeric_answer  # noqa: E402
from inference_common import normalize_answer  # noqa: E402
from utils import llama_gsm8k_parse, llama_mbpp_parse  # noqa: E402

QA = [
    ("Paris", "paris"),
    ("The capital of France is Paris.", "Paris"),
    ("", "Paris"),
    ("Paris", ""),
    ("PARIS, France.", "paris france"),
    ("dog", "cat"),
    ("aaa", "a"),
    ("The Eiffel Tower is in Paris", "eiffel tower paris"),
]
GSM = [
    ("The final answer is 42.", ["42"]),
    ("the final answer is $1,234", ["1234"]),
    ("The final answer is 3.50", ["3.5"]),
    ("blah blah no answer here", ["5"]),
    ("The final answer is -7", ["-7"]),
    ("Final answer is 100", ["100", "200"]),
    ("The final answer is 99", ["42"]),
    ("So the final answer is 5,000.", ["5000"]),
]
MBPP_RAW = [
    "```python\ndef f():\n    return 1\n```",
    "Here is the code:\ndef g(x):\n    return x*2\n```",
    "<|python_tag|>```python\nx = 1\n```<|eot_id|>",
    "no code at all",
]
GSM_RAW = [
    "The final answer is 42.",
    "the FINAL answer IS $1,234",
    "I think the final answer is -3.5 maybe",
    "no answer",
]
NORM = ["Hello, World!", "  THE   dog  ", "$1,234.00", "3.50", "-7", "", "abc"]

golden = {
    "qa": [{"pred": p, "ref": r, **compute_metrics(p, r)} for p, r in QA],
    "gsm": [{"pred": p, "answers": a, "result": compute_gsm8k_metrics(p, a)} for p, a in GSM],
    "mbpp_parse": [{"raw": s, "out": llama_mbpp_parse(s)} for s in MBPP_RAW],
    "gsm_parse": [{"raw": s, "out": llama_gsm8k_parse(s)} for s in GSM_RAW],
    "norm": [{"s": s, "answer": normalize_answer(s), "numeric": normalize_numeric_answer(s)} for s in NORM],
}
out = pathlib.Path(__file__).parent / "_golden" / "eval_golden.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(golden, out)
print(
    f"wrote {out} | qa={len(QA)} gsm={len(GSM)} mbpp_parse={len(MBPP_RAW)} "
    f"gsm_parse={len(GSM_RAW)} norm={len(NORM)}"
)
print("sample qa[1]:", golden["qa"][1])
print("sample gsm[2].result:", golden["gsm"][2]["result"])
