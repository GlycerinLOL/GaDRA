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

All machine-specific settings (repo path, account/partition, secrets) live in **one sourced file**, so you
never edit the committed `.slurm` scripts. Copy the template, fill it in, and keep it out of git:

```bash
# Clone onto the shared filesystem (visible from login + compute nodes).
git clone https://github.com/GlycerinLOL/GaDRA /path/to/shared/GaDRA
cd /path/to/shared/GaDRA

# Create your machine-local env file (gitignored) from the template, then edit it.
cp examples/slurm/env.local.sh.example examples/slurm/env.local.sh
$EDITOR examples/slurm/env.local.sh        # set GADRA_REPO, SBATCH_ACCOUNT/PARTITION, OPENAI_API_KEY, ...

# Source it, then build the pinned env (cu124 torch + prebuilt flash-attn + deepspeed; no compilation).
source examples/slurm/env.local.sh
uv sync --group gpu --locked

# Cache your HF token (the CLI lives in the uv env) and pre-download the gated base model while online.
uv run huggingface-cli login
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct')"
```

From then on, **every session** you just `source` the env file before submitting (or add the `source` line to
your `~/.bashrc`):

```bash
source examples/slurm/env.local.sh
sbatch examples/slurm/train.slurm          # or examples/slurm/inference.slurm
```

`sbatch` forwards the submitting shell's environment to the job, so one `source` covers both the scheduler
knobs (`SBATCH_ACCOUNT` / `SBATCH_PARTITION`) and the runtime vars (`GADRA_REPO`, `UV_*`, `OPENAI_API_KEY`,
`CONFIG`, `ADAPTER`). The scripts resolve the repo root from `GADRA_REPO` → the `sbatch` dir → `$PWD`.

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
- **Method / variants**: set the `METHOD` env var — `METHOD=gadra-mono` (GaDRA-Mono), `METHOD=gadra-soft`
  (GaDRA-Soft), or `METHOD=lora` (the LoRA baseline) — e.g. `METHOD=gadra-mono sbatch examples/slurm/train.slurm`.
  It overrides the config's `method:` field, so one wrapper trains any shipped method (no file edit). The
  save-dir label is the config's `exp_name:` field — set it alongside `method:` to keep results tidy, or set the
  `EXP_NAME` env var to override it without editing the config. Use a different `CONFIG=...` to point at another
  run-config entirely. **MoE / block-parallel**: `METHOD=gadra-parallel` (gated) or `METHOD=lora-parallel`
  (ungated baseline) attach the adapter to a whole MLP/MoE block — submit with
  `CONFIG=examples/config/train_olmoe.yaml sbatch examples/slurm/train.slurm` (set `parallel_modules:` to the
  backbone's block name there: OLMoE/Mixtral `mlp`, Granite/Phi `block_sparse_moe`, Llama4 `feed_forward`).
- **Weights & Biases** (off by default): set `report_to: wandb` in [`train.yaml`](../config/train.yaml) and
  either fill `wandb_project` / `wandb_entity` there or export `WANDB_PROJECT` / `WANDB_ENTITY` before
  `sbatch` (they forward to the job). Run `uv run wandb login` once on the login node first.
- **Output dir** `save/<DATASET>/<DATE>/CPT/<exp_name>/`, built by the script. The last component comes from
  the config's `exp_name:` field (`GaDRA` by default); `DATASET=BBC_news` and `DATE=$(date +%Y%m%d)` are env
  knobs, and `EXP_NAME` overrides the config's value (e.g. `EXP_NAME=GaDRA-Mono METHOD=gadra-mono sbatch …`).
  A re-run with the same dataset+date+exp_name **overwrites** it (clean slate).
- **GPU by default** — the script requests `--gpus-per-node=2` and trains on GPU (DeepSpeed `use_cpu: false`,
  one process per GPU); no extra step needed. Change the count with `sbatch --gpus-per-node=N`.

## 3. Submit inference

**One job evaluates all the benchmarks listed in the config against one adapter**, loads the model **once**,
and prints a score summary to its `.out` log. The default config
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

### Per-task settings

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
  this; the `out/gadra-bbc/` in `train.yaml` is the default for a manual, non-SLURM run). A full copy of the
  run log is `tee`'d into that dir as **`log.txt`** (run-config, model structure, sample rows, the Trainer's
  `***** Running training *****` summary, per-step losses); `slurm-logs/<jobid>.{out,err}` hold the same stream.
- **Inference** streams per-batch progress (`[gsm8k] generated 320/1319`), then a per-checkpoint summary to
  the `.out` log, and **saves results to disk**. When `adapter` points at an
  experiment ROOT, **every `checkpoint-N/` and the final adapter are evaluated in order** (model reloaded per
  checkpoint); point it at a single `checkpoint-N/` to run just one. Each (checkpoint, task) writes:
  ```
  <checkpoint>/inference_results/<task>/distribution_stats.json   # aggregate metric (first place to look)
  <checkpoint>/inference_results/<task>/inference_result.json     # per-sample predictions / references
  ```
  Summary block in the log:
  ```
  ===== summary (checkpoint=final, adapter=save/BBC_news/<DATE>/CPT/GaDRA) =====
    BBC QA   Correct=83.50% over 400 samples (GPT judge: gpt-4.1-mini)
    GSM8K    EM=74.30 over 1319 samples
    MBPP     pass@1=52.40 over 500 samples
    TiEBe    Correct=41.20% over ... samples (GPT judge: gpt-4.1-mini)
  ```
  (Numbers illustrative.) A failed task shows `FAILED: <reason>` and the job exits non-zero, but the rest run.
  Greedy decoding uses the **dynamic KV cache by default** (fast, and bit-identical to the static-cache
  output); set `cache_implementation: static` in a per-task config only for the verbatim (slower) knobs.

## Data

Data is **not** shipped. For SLURM the files must live on the **shared filesystem** (readable from the offline
compute nodes); the simplest place is a gitignored `data/` dir inside `$GADRA_REPO`. Set `train_file` /
`eval_file` (+ `documents_file` for `bbcqa`) in the configs, or `--override` them. Formats + fields:
[`data/README.md`](../../data/README.md).

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
