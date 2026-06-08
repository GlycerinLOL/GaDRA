#!/bin/bash
# GaDRA paper-repro inference / eval under SLURM (uv). Single GPU, config-driven.
# See examples/slurm/README.md for setup, knobs, and submission examples.

#SBATCH --job-name=gadra-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err

set -euo pipefail

# --- Repo root ---
REPO="${GADRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -e "${REPO}/pyproject.toml" ]; then
    echo "ERROR: GADRA_REPO='${REPO}' is not the GaDRA repo root (no pyproject.toml there)." >&2
    echo "       Set GADRA_REPO (e.g. via examples/slurm/env.local.sh), or submit from the repo root." >&2
    exit 2
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO/.uv-python}"
export UV_LINK_MODE=copy          # copy mode (cross-FS safe)
export UV_NO_SYNC=1               # hermetic: no network / lock mutation from a job
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HF_ALLOW_CODE_EVAL=1       # required for MBPP pass@1; harmless for other tasks
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# OPENAI_API_KEY is never stored here — bbcqa/tiebe read it from the environment.
# `export OPENAI_API_KEY=sk-...` before sbatch; it propagates to the job automatically.

# --- Isolate from conda ---
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true

cd "$REPO"
mkdir -p slurm-logs

# --- Config-driven eval ---
CONFIG="${CONFIG:-examples/config/inference.yaml}"
OVERRIDES=()
[ -n "${ADAPTER:-}" ] && OVERRIDES+=(--override "adapter=${ADAPTER}")
echo "config: ${CONFIG}${ADAPTER:+  adapter=${ADAPTER}}"
uv run --no-sync --offline python -m examples.inference \
    --config "${CONFIG}" ${OVERRIDES[@]+"${OVERRIDES[@]}"}
