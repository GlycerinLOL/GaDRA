# GaDRA reproduction tooling (`examples/`)

**This directory is repo-only — it is NOT part of the `pip install gadra` package.** The installable
package (`src/gadra`) is the pure `peft` method. Everything here is the tooling that reproduces the paper:
the parity-exact data pipeline, the deterministic scorers, the legacy-checkpoint converter, and two
config-driven entry scripts. It lives outside `src/`, so it is never built into the wheel.

## Install (uv)

```bash
uv sync --group gpu          # cu124 torch + prebuilt flash-attn + deepspeed + datasets/evaluate/accelerate/pyyaml
```

See the repo [README → Reproducing the paper](../README.md#reproducing-the-paper) for prerequisites
(NVIDIA driver ≥ 525.60.13) and the uv install.

## Run (config-driven, from the repo root)

```bash
# Training — the adapter is chosen by the config's `method:` field (default: gadra)
uv run python -m examples.train --config examples/config/train.yaml
#   variants: --override method=gadra-mono | --override method=gadra-soft | --override method=lora

# Inference / paper-repro eval (task: qa | gsm8k | mbpp; deterministic metrics)
uv run python -m examples.inference --config examples/config/inference.yaml
```

Multi-GPU (accelerate + DeepSpeed ZeRO-2) and GPT-judged eval (`bbcqa` / `tiebe`) are in the repo
[README → Reproducing the paper](../README.md#reproducing-the-paper).

**Data is not shipped.** The configs point at `data/*.jsonl` placeholders — supply your own and edit the
config (or `--override train_file=... eval_file=...`). See the repo
[README → Data](../README.md#data-you-provide-it) for the JSONL formats, fields, and where to put the files.

Override any config field on the CLI (YAML-typed): `--override learning_rate=5e-4 --override packing=group`.
The scripts also tolerate a direct `python examples/<script>.py` invocation (a small `sys.path` shim adds
the repo root) so SLURM file-path wrappers keep working.

## Layout

| Path | What |
|---|---|
| `train.py` | single training entry for **all** methods — the adapter is chosen by the config's `method:` field (gadra \| gadra-mono \| gadra-soft \| lora) via an in-file method registry (add a baseline = one `@register`); standard `transformers.Trainer` |
| `inference.py` | paper-repro eval entry — reads `config/inference.yaml`; `task: qa \| gsm8k \| mbpp` (deterministic) + `bbcqa \| tiebe` (GPT-judged Correct%) |
| `processing.py` | data: tokenizer + EOS + FA2 `PackingCollator` (golden-tested, G2) |
| `evaluation.py` | parsers + deterministic scorers (QA F1/EM, GSM8K EM, MBPP pass@1) + greedy runner + PPL (G3) + `GPTJudge` (BBC-QA / TiEBe Correct%, key from `OPENAI_API_KEY` env, not bit-exact) |
| `convert.py` | one-time legacy → peft-native checkpoint converter (`from examples.convert import convert_checkpoint`) |
| `config/train.yaml` · `config/inference.yaml` | run-configs |
| `config/deepspeed_zero2.yaml` | DeepSpeed ZeRO-2 for multi-GPU (copied verbatim from the paper's setup) |
| `config/llama3.2-Instruct.jinja` | chat template (`--override chat_template_path=...`) |
| `slurm/` | uv SLURM wrappers + a [submission walkthrough](slurm/README.md) (offline two-phase; train = multi-GPU accelerate+ZeRO-2, inference = single-GPU eval) |
| `tests/` | the processing / evaluation / convert parity tests + their goldens (`pytest examples/tests/`) |

## Data packing

`config/train.yaml`'s `packing:` selects the strategy:
- `fa2_collator` (default, paper-faithful) — per-document varlen packing; REQUIRES flash-attention-2 (in the `gpu` group).
- `group` (portable) — concatenate-and-chunk under SDPA; no flash-attn (not bit-exact to the paper). Use `--override packing=group`.
