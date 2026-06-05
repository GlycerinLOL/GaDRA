"""Capture the G2 data-path golden from the research repo's ``Preprocessing`` (maintainer-only).

Freezes the parent pipeline's output on a few real BBC pretraining samples so ``test_processing.py`` can
assert ``examples.processing`` reproduces it bit-for-bit. Run once on a host where the original repo is importable::

    GADRA_SOURCE_REPO=/path/to/Adapter_Research python GaDRA/examples/tests/capture_data_golden.py

Writes ``examples/tests/_golden/data_golden.pt`` (committed). Skips cleanly when the source repo is absent.
"""

import json
import os
import pathlib
import sys

import torch

REPO = os.environ.get("GADRA_SOURCE_REPO")
if not REPO or not pathlib.Path(REPO).is_dir():
    sys.exit("capture_data_golden: set GADRA_SOURCE_REPO to the research-repo root (maintainer-only).")
sys.path.insert(0, REPO)

N = int(os.environ.get("N_SAMPLES", "8"))
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
TOK_DIR = os.environ.get("TOKENIZER_DIR", os.path.join(REPO, "save/BBC_news/gadra_pkg_repro_oldfmt/CPT/GaDRA_full"))
CORPUS = os.environ.get("CORPUS", os.path.join(REPO, "datasets/BBC_news_alltime/Pretraining/handwritten_5587.jsonl"))
CHAT_TEMPLATE = os.path.join(REPO, "configs/llama3.2-Instruct.jinja")

from Preprocessing import PackingCollator, Preprocessor  # noqa: E402  (research repo)
from utils import check_tokenizer_auto_adds_eos, get_first_sample_text, get_tokenizer_for_preprocess  # noqa: E402

texts = []
with open(CORPUS, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        if i >= N:
            break
        d = json.loads(line)
        texts.append(d.get("text") or d.get("content"))

tokenizer = get_tokenizer_for_preprocess(
    model_name_or_path=TOK_DIR,
    chat_template_path=CHAT_TEMPLATE,
    truncation_side="right",
    padding_side="right",
)
auto_add_eos = check_tokenizer_auto_adds_eos(tokenizer, get_first_sample_text([{"text": t} for t in texts]))

pre = Preprocessor(training_mode="pt", preprocess_strategy="packing", auto_add_eos=auto_add_eos)
batched = pre.tokenize_function({"text": texts}, tokenizer=tokenizer, max_length=MAX_LEN)
sample_features = [
    {"input_ids": list(batched["input_ids"][i]), "labels": list(batched["labels"][i])}
    for i in range(len(texts))
]
packed = PackingCollator()(sample_features)
grouped = pre.grouping_concate_all(
    {"input_ids": [f["input_ids"] for f in sample_features], "labels": [f["labels"] for f in sample_features]},
    block_size=MAX_LEN,
)

golden = {
    "meta": {
        "n": len(texts),
        "max_len": MAX_LEN,
        "tokenizer_dir": TOK_DIR,
        "auto_add_eos": bool(auto_add_eos),
        "eos_token": tokenizer.eos_token,
    },
    "texts": texts,
    "sample_features": sample_features,
    "packed": {k: v.tolist() for k, v in packed.items()},
    "grouped": {"input_ids": grouped["input_ids"], "labels": grouped["labels"]},
}
out = pathlib.Path(__file__).parent / "_golden" / "data_golden.pt"
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(golden, out)
print(
    f"wrote {out} | samples={len(texts)} | auto_add_eos={auto_add_eos} "
    f"| per-sample lens={[len(f['input_ids']) for f in sample_features]} "
    f"| packed_len={len(golden['packed']['input_ids'][0])} | grouped_blocks={len(grouped['input_ids'])}"
)
