# `data/`

This directory is where you put the **datasets you provide** for reproduction — GaDRA ships **no** data.

Everything in here is **gitignored except this README**, so your corpora / eval files are never committed.
(For SLURM, this directory must live on the shared filesystem so the offline compute nodes can read it.)

## What goes here

| File (default path in the configs) | Used by | Format — one JSON object per line |
|---|---|---|
| `corpus.jsonl` | CPT **train** — `train_file` in [`../examples/config/train.yaml`](../examples/config/train.yaml) | `{"text": "..."}` — raw continual-pretraining text |
| `valid.jsonl` | CPT **validation** (optional) — `validation_file` in train.yaml; enables eval during training | same `{"text": "..."}` format |
| `gsm8k_test.jsonl`, `bbcqa_test.jsonl`, … | eval — `eval_file` in [`../examples/config/inference.yaml`](../examples/config/inference.yaml) | the paper's raw eval files work as-is |
| _(bbcqa)_ a documents JSONL | eval — `documents_file` in `inference.yaml` (the GPT judge's source articles, keyed by `id`) | `{"id": ..., "text": ...}` |

Keep the default file names above, or point the config at your paths
(`--override train_file=data/<your>.jsonl` / `--override eval_file=data/<your>.jsonl`).

Per-task eval fields (`qa` / `gsm8k` / `mbpp` / `bbcqa` / `tiebe`) are documented inline in
[`../examples/config/inference.yaml`](../examples/config/inference.yaml).
