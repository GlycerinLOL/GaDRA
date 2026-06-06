#!/bin/bash
# GaDRA paper-repro inference / eval under SLURM, managed with uv (no conda).
#
# uv counterpart of the research repo's conda inference launcher. Single GPU, one process
# (the model is sharded by device_map="auto"); no accelerate / DeepSpeed (eval is single-GPU).
#
# TWO-PHASE workflow (compute nodes are typically offline) — same env as examples/slurm/train.slurm:
#   1) ON THE LOGIN NODE (has internet), ONCE, before submitting — populate the shared uv cache + .venv:
#        export GADRA_REPO=/path/to/shared/GaDRA   # your checkout, on a filesystem both node types can see
#        export UV_CACHE_DIR="$GADRA_REPO/.uv-cache"
#        export UV_PYTHON_INSTALL_DIR="$GADRA_REPO/.uv-python"
#        cd "$GADRA_REPO" && uv sync --group gpu --locked
#   2) In examples/config/inference.yaml set `adapter` ONCE (the suite runs all tasks in its `tasks:` list);
#      in each examples/config/inference_<task>.yaml set `eval_file` (+ documents_file for bbcqa) to your data.
#
# Submit — DEFAULT runs the whole suite (BBC QA + GSM8K + MBPP + TiEBe) in one job, model loaded once:
#   export OPENAI_API_KEY=sk-...                                # suite includes GPT-judged bbcqa + tiebe
#   sbatch examples/slurm/inference.slurm
# Single benchmark instead — CONFIG selects a per-task config; ADAPTER points it at your save dir (no edits):
#   ADAPTER=save/BBC_news/<DATE>/CPT/GaDRA CONFIG=examples/config/inference_gsm8k.yaml sbatch examples/slurm/inference.slurm  # GSM8K (no key)
#   ADAPTER=save/BBC_news/<DATE>/CPT/GaDRA CONFIG=examples/config/inference_bbcqa.yaml sbatch examples/slurm/inference.slurm  # BBC QA (needs key)
# Per-task knobs (use_raw_text / max_new_tokens / GPT judge) live in the per-task configs; see
# examples/slurm/README.md -> "Per-task settings". On an account/partition cluster (e.g. nano5), also set once:
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>

#SBATCH --job-name=gadra-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err

set -euo pipefail

# --- Repo root: GADRA_REPO wins; else the dir you ran `sbatch` from (SLURM_SUBMIT_DIR), else $PWD. Set
# machine-specific vars in a sourced env file rather than editing this script — see examples/slurm/env.local.sh.example ---
REPO="${GADRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -e "${REPO}/pyproject.toml" ]; then
    echo "ERROR: GADRA_REPO='${REPO}' is not the GaDRA repo root (no pyproject.toml there)." >&2
    echo "       Set GADRA_REPO (e.g. via examples/slurm/env.local.sh), or submit from the repo root." >&2
    exit 2
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO/.uv-python}"
# cache and .venv may sit on different mounts on HPC -> copy instead of hardlink to avoid cross-FS failures.
export UV_LINK_MODE=copy
# Hermetic: never re-resolve / touch the network / mutate the committed lock from a job.
export UV_NO_SYNC=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MBPP runs candidate code in a sandbox; other tasks ignore it. Harmless to leave on.
export HF_ALLOW_CODE_EVAL=1
export TOKENIZERS_PARALLELISM=false

# OPENAI_API_KEY is NEVER stored here. bbcqa / tiebe (GPT judge) read it from the environment;
# `export OPENAI_API_KEY=...` before sbatch and it propagates (sbatch exports the submitting env by default).
# Deterministic tasks (qa / gsm8k / mbpp) need no key.

# --- Isolate from conda (the account keeps conda for the parent research repo; do NOT let it bleed in) ---
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true

cd "$REPO"
mkdir -p slurm-logs

# Config-driven eval — CONFIG picks the run-config (default = the inference.yaml suite, which has `adapter`
# set once). For a single per-task config (no `tasks:` list, no `adapter:`), point the adapter via ADAPTER=...
# without editing any file. ADAPTER also overrides the suite's adapter when set.
CONFIG="${CONFIG:-examples/config/inference.yaml}"
OVERRIDES=()
[ -n "${ADAPTER:-}" ] && OVERRIDES+=(--override "adapter=${ADAPTER}")
echo "config: ${CONFIG}${ADAPTER:+  adapter=${ADAPTER}}"
uv run --no-sync --offline python -m examples.inference \
    --config "${CONFIG}" ${OVERRIDES[@]+"${OVERRIDES[@]}"}
