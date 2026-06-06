# Running GaDRA on SLURM (uv)

End-to-end workflow for submitting GaDRA jobs on an HPC cluster with [uv](https://docs.astral.sh/uv/) — no
conda. Two wrappers live here:

| Script | Job | GPUs |
|---|---|---|
| [`train.slurm`](train.slurm) | continual pre-training (CPT) | multi-GPU (accelerate + DeepSpeed ZeRO-2) |
| [`inference.slurm`](inference.slurm) | paper-repro eval | single-GPU |

> **Why two phases?** HPC compute nodes are usually **offline** (no internet). You build the uv environment
> **once on the login node** (which has internet) onto a **shared filesystem**, and every job then runs that
> env **offline**. The scripts already export `UV_NO_SYNC=1` / `HF_HUB_OFFLINE=1` for this.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed on the login node (`curl -LsSf https://astral.sh/uv/install.sh | sh`,
  then `source "$HOME/.local/bin/env"` or restart the shell).
- An NVIDIA GPU partition with **driver ≥ 525.60.13** (the env uses the bundled CUDA 12.4 runtime).
- A **shared filesystem** path visible from BOTH login and compute nodes (for the repo + uv cache + `.venv`).
- Hugging Face access to the gated `meta-llama/Llama-3.1-8B-Instruct`.
- Your training / eval **data** (not shipped — see [Data](#data)).

## 1. One-time setup (login node — has internet)

```bash
# Clone onto the shared filesystem and point uv's cache + python there (so compute nodes can reuse them).
git clone https://github.com/GlycerinLOL/GaDRA /path/to/shared/GaDRA
export GADRA_REPO=/path/to/shared/GaDRA
export UV_CACHE_DIR="$GADRA_REPO/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$GADRA_REPO/.uv-python"

# Build the pinned env (cu124 torch + prebuilt flash-attn + deepspeed; no compilation). Downloads everything
# into the cache above; commits nothing.
cd "$GADRA_REPO" && uv sync --group gpu --locked

# Cache your HF token (the CLI lives in the uv env) and pre-download the gated base model while online.
uv run huggingface-cli login
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct')"
```

These three exports (`GADRA_REPO`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`) must be **set in the shell you
submit from** — `sbatch` forwards the submitting environment to the job by default, which is how the offline
compute node finds the login-node env. Put them in your `~/.bashrc` (or re-export them each session).

On an **account/partition** cluster (e.g. nano5), also set these once — Slurm reads them, no file edit:

```bash
export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
```

## 2. Submit training

```bash
cd "$GADRA_REPO"

# Default (uses the GPUs the script requests — 2):
sbatch examples/slurm/train.slurm

# N GPUs: grad-accum auto-adjusts so the paper's global batch size (128) is held.
sbatch --gpus-per-node=4 examples/slurm/train.slurm
```

- **Knobs** (env vars, paper defaults): `N_GPUS` (auto-detected from the allocation), `MBS=16`
  (`per_device_train_batch_size`), `GBS=128` (target global batch). The script computes
  `grad_accum = GBS / (MBS × N_GPUS)` and prints the effective global batch — check the `.out` log:
  `GPUs=4  per_device_bs=16  grad_accum=2  ->  global_batch=128 (target 128)`.
- **The recipe** (LR, warmup, epochs, r=512, dual/hard, …) lives in
  [`examples/config/train.yaml`](../config/train.yaml). Edit it, or override at the bottom of the script.
- **Variants**: add `--override router_conditioning=mono` (GaDRA-Mono) or `--override gate=soft` to the
  `examples.train` line in the script.
- **Weights & Biases** (off by default): set `report_to: wandb` in [`train.yaml`](../config/train.yaml) and
  either fill `wandb_project` / `wandb_entity` there or export `WANDB_PROJECT` / `WANDB_ENTITY` before
  `sbatch` (they forward to the job). Run `uv run wandb login` once on the login node first.
- **Output dir** mirrors the research repo's layout — `save/<DATASET>/<DATE>/CPT/<EXP_NAME>/`, built by the
  script (knobs `DATASET=BBC_news`, `EXP_NAME=GaDRA`, `DATE=$(date +%Y%m%d)`; e.g. `export EXP_NAME=GaDRA-Mono`
  before `sbatch`). A re-run with the same dataset+date+exp_name **overwrites** it (clean slate, like the original).
- **GPU by default** — the script requests `--gpus-per-node=2` and trains on GPU (DeepSpeed `use_cpu: false`,
  one process per GPU); no extra step needed. Change the count with `sbatch --gpus-per-node=N`.

## 3. Submit inference

**One job evaluates all the benchmarks listed in the config against one adapter** (like the research repo's
`task_types` list), loads the model **once**, and prints a score summary to its `.out` log. The default config
[`examples/config/inference.yaml`](../config/inference.yaml) is a **suite** that runs BBC QA + GSM8K + MBPP +
TiEBe — you set the adapter **once** there:

```bash
cd "$GADRA_REPO"

# One-time edits:
#   1) examples/config/inference.yaml   -> set `adapter:` to your save dir (e.g. save/BBC_news/<DATE>/CPT/GaDRA)
#      (and add/remove task lines in `tasks:` to choose which benchmarks run this job)
#   2) examples/config/inference_<task>.yaml -> set each `eval_file:` (+ `documents_file:` for bbcqa) to your data

export OPENAI_API_KEY=sk-...        # needed because the suite includes the GPT-judged bbcqa + tiebe; never commit it
sbatch examples/slurm/inference.slurm                 # runs the whole suite (4 tasks) in one job
```

A failing task (e.g. a missing key or file) is reported in the summary and **does not abort the others**; the
job exits non-zero if any task failed. `HF_ALLOW_CODE_EVAL=1` (MBPP) is already exported by the script.

**Run a single benchmark** instead — point `CONFIG` at one per-task config and `ADAPTER` at your save dir
(per-task configs carry no model path, so `ADAPTER` supplies it — no file edit):

```bash
ADAPTER=save/BBC_news/<DATE>/CPT/GaDRA CONFIG=examples/config/inference_gsm8k.yaml sbatch examples/slurm/inference.slurm   # GSM8K (no key)
ADAPTER=save/BBC_news/<DATE>/CPT/GaDRA CONFIG=examples/config/inference_bbcqa.yaml sbatch examples/slurm/inference.slurm   # BBC QA (needs key)
```

`ADAPTER` (when set) also overrides the suite's adapter for all tasks. Or run fewer tasks by trimming the
`tasks:` list in `inference.yaml` — the adapter still lives in exactly one place there.

### Per-task settings (matches the research repo's `configs/eval_config/*.json`)

The suite stamps the shared `adapter` onto every task; each per-task config supplies its own eval file and the
correct knobs. This table is the mapping (and the three footguns to avoid if you hand-edit a config):

| Paper benchmark | `task` | original eval file | `use_raw_text` | `chat_template_path` | `max_new_tokens` | key |
|---|---|---|---|---|---|---|
| **BBC QA** (target) | `bbcqa` ⚠ | `…/v2_5587-test400.jsonl` (+ `handwritten_5587.jsonl` as `documents_file`) | false | llama jinja | 256 | yes |
| GSM8K (retention) | `gsm8k` | `datasets/GSM8K/gsm8k_llama_test.jsonl` | **true** (8-shot) ⚠ | **null** | 256 | no |
| MBPP (retention) | `mbpp` | `datasets/MBPP/mbpp_llama_test.jsonl` | **true** (3-shot) ⚠ | **null** | 256 | no |
| TiEBe (retention) | `tiebe` | `datasets/TiEBe/pre_2023/world.jsonl` | false | llama jinja | **128** ⚠ | yes |

Footguns: (1) **BBC QA is GPT-judged** — use `task=bbcqa`, not `task=qa` (that's char-F1/EM, a different
metric). (2) **GSM8K/MBPP are few-shot via the raw `text` field** — they need `use_raw_text=true` (else they
silently degrade to 0-shot chat). (3) **TiEBe uses `max_new_tokens=128`**, not 256.

Generation matches the original bit-for-bit (greedy, `eos=[eos, <|eot_id|>]`, left-padding), so deterministic
tasks (GSM8K/MBPP and char-F1 QA) are reproducible exactly. `batch_size` is perf-only (greedy is
batch-invariant). Not covered by these wrappers: CCQA (zh target), commonsense_multi, and the Qwen3/OLMoE
backbone variants — open an issue if you need them.

## Monitor & outputs

```bash
squeue --me                                  # your queued/running jobs
tail -f slurm-logs/<jobid>.out               # live stdout (the grad-accum / progress lines)
tail -f slurm-logs/<jobid>.err               # errors
```

- **Training** writes the adapter + tokenizer to `save/<DATASET>/<DATE>/CPT/<EXP_NAME>/` (the slurm job sets
  this; the `out/gadra-bbc/` in `train.yaml` is only the default for a manual, non-SLURM run). A full copy of
  the run log is `tee`'d into that dir as **`log.txt`** (same as the research repo): the resolved run-config,
  the model structure, packed-example counts + sample rows, the Trainer's `***** Running training *****`
  summary (num examples, per-device / global batch size, grad-accum, optimization steps, trainable params),
  and the per-step losses. The `slurm-logs/<jobid>.{out,err}` files hold the same stream (plus launcher noise).
- **Inference** prints each task's metric as it finishes, then a final summary block to the `.out` log:
  ```
  ===== GaDRA inference summary (adapter=save/BBC_news/<DATE>/CPT/GaDRA) =====
    BBC QA   Correct=83.50% over 400 samples (GPT judge: gpt-4.1-mini)
    GSM8K    EM=74.30 over 1319 samples
    MBPP     pass@1=52.40 over 500 samples
    TiEBe    Correct=41.20% over ... samples (GPT judge: gpt-4.1-mini)
  ```
  (Numbers above are illustrative.) A task that fails shows `FAILED: <reason>` in the summary and the job
  exits non-zero, but the other tasks still run.

## Data

Data is **not** shipped — you provide it. For SLURM the files must live on the **shared filesystem** (readable
from the offline compute nodes); the simplest place is a `data/` dir inside `$GADRA_REPO` (it's gitignored), and
config paths are relative to the repo root.

- **CPT** — set `train_file` (+ optional `validation_file`) in [`../config/train.yaml`](../config/train.yaml);
  format: one `{"text": "..."}` per line.
- **Inference** — set `eval_file` (+ `documents_file` for `bbcqa`) in
  [`../config/inference.yaml`](../config/inference.yaml); the paper's raw files work directly.

Set the paths by editing the YAML, or add `--override train_file=...` / `--override eval_file=...` to the
`examples.train` / `examples.inference` line in the script. Full format spec:
[repo README → Data](../../README.md#data-you-provide-it).

## Troubleshooting

- **`uv: command not found`** at submit time → `source "$HOME/.local/bin/env"` (or restart the shell) on the
  login node before `sbatch`.
- **Job can't reach the network / model** → you skipped the login-node pre-sync, or `GADRA_REPO` /
  `UV_CACHE_DIR` differ between the pre-sync and the job. Re-run step 1 with the exact same exports.
- **`Invalid account/partition`** → set `SBATCH_ACCOUNT` / `SBATCH_PARTITION` (step 1).
- **Effective global batch ≠ 128** (warning in the log) → your GPU count doesn't divide 128 by `MBS`; pick a
  power-of-two GPU count or set `MBS` / `GBS` accordingly.
- More: the repo [FAQ](../../docs/FAQ.md).

> **Multi-node** (across machines) is out of scope for these single-node wrappers. Open an issue if you need
> cross-node rendezvous (`main_process_ip` / `machine_rank` / `srun`).
