# GaDRA reproduction tooling (`examples/`)

**This directory is repo-only — it is NOT part of the `pip install gadra` package.** The installable
package (`src/gadra`) is the pure `peft` method. Everything here is the tooling that reproduces the paper:
the parity-exact data pipeline, the deterministic scorers, the legacy-checkpoint converter, and two
config-driven entry scripts. It lives outside `src/`, so it is never built into the wheel.

## Install (uv)

```bash
uv sync --group gpu          # cu124 torch + prebuilt flash-attn + deepspeed + datasets/evaluate/accelerate/pyyaml
```

See the repo [README → Installation](../README.md#installation) for prerequisites
(NVIDIA driver ≥ 525.60.13) and the uv install.

## Run (config-driven, from the repo root)

```bash
# Training — the adapter is chosen by the config's `method:` field (default: gadra)
uv run python -m examples.train --config examples/config/train.yaml
#   variants: --override method=gadra-mono | --override method=gadra-soft | --override method=lora

# Inference / paper-repro eval (task: qa | gsm8k | mbpp; deterministic metrics)
uv run python -m examples.inference --config examples/config/inference.yaml
```

Multi-GPU (accelerate + DeepSpeed ZeRO-2):

```bash
# For N GPUs, hold the paper's global batch size 128: gradient_accumulation_steps = 128 / (16 * N)
# (the SLURM wrapper computes this automatically).
uv run accelerate launch --num_processes <N_GPUS> \
    --config_file examples/config/deepspeed_zero2.yaml \
    -m examples.train --config examples/config/train.yaml
```

GPT-judged eval (`bbcqa` / `tiebe`, "Correct %") — the key is read from the environment, never the repo:

```bash
export OPENAI_API_KEY=sk-...
uv run python -m examples.inference --config examples/config/inference.yaml --override task=bbcqa
```

The eval derives each sample's prompt / reference / judge inputs exactly as the research harness does, so the
paper's raw eval files work directly: the deterministic metrics (QA char-F1/EM, GSM8K EM, MBPP pass@1) reproduce
bit-for-bit; `bbcqa` / `tiebe` use the verbatim GPT judge and are method-equivalent (they call OpenAI). Which
numbers are bit-exact: [`../docs/FAQ.md`](../docs/FAQ.md).

**SLURM** (offline compute nodes): set `GADRA_REPO`, pre-sync once on the login node
(`uv sync --group gpu --locked`), then `sbatch examples/slurm/{train,inference}.slurm` — full walkthrough in
[`slurm/README.md`](slurm/README.md). On an account/partition cluster set
`export SBATCH_ACCOUNT=... SBATCH_PARTITION=...` once.

**Data is not shipped.** The configs point at `data/*.jsonl` placeholders — supply your own and edit the
config (or `--override train_file=... eval_file=...`). See [`../data/README.md`](../data/README.md) for the
JSONL formats, fields, and where to put the files.

Override any config field on the CLI (YAML-typed): `--override learning_rate=5e-4 --override packing=group`.
The scripts also tolerate a direct `python examples/<script>.py` invocation (a small `sys.path` shim adds
the repo root) so SLURM file-path wrappers keep working.

## Layout

| Path | What |
|---|---|
| `train.py` | single training entry for **all** methods — the adapter is chosen by the config's `method:` field (gadra \| gadra-mono \| gadra-soft \| lora) via an in-file method registry (add a baseline = one `@register`); standard `transformers.Trainer` |
| `inference.py` | paper-repro eval entry — reads `config/inference.yaml`; `task: qa \| gsm8k \| mbpp` (deterministic) + `bbcqa \| tiebe` (GPT-judged Correct%) |
| `processing.py` | data: tokenizer + EOS + FA2 `PackingCollator` (golden-tested) |
| `evaluation.py` | parsers + deterministic scorers (QA F1/EM, GSM8K EM, MBPP pass@1) + greedy runner + PPL + `GPTJudge` (BBC-QA / TiEBe Correct%, key from `OPENAI_API_KEY` env, not bit-exact) |
| `convert.py` | one-time legacy → peft-native checkpoint converter (`from examples.convert import convert_checkpoint`) |
| `config/train.yaml` · `config/inference.yaml` | run-configs |
| `config/deepspeed_zero2.yaml` | DeepSpeed ZeRO-2 for multi-GPU |
| `config/llama3.2-Instruct.jinja` | chat template (`--override chat_template_path=...`) |
| `slurm/` | uv SLURM wrappers + a [submission walkthrough](slurm/README.md) (offline two-phase; train = multi-GPU accelerate+ZeRO-2, inference = single-GPU eval) |
| `tests/` | the processing / evaluation / convert parity tests + their goldens (`pytest examples/tests/`) |

## Data packing

`config/train.yaml`'s `packing:` selects the strategy:
- `fa2_collator` (default, paper-faithful) — per-document varlen packing; REQUIRES flash-attention-2 (in the `gpu` group).
- `group` (portable) — concatenate-and-chunk under SDPA; no flash-attn (not bit-exact to the paper). Use `--override packing=group`.
