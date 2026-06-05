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
#   2) Edit examples/config/inference.yaml (task / adapter / eval_file), then submit this script.
#
# Submit:
#   sbatch examples/slurm/inference.slurm                              # deterministic task (qa / gsm8k / mbpp)
#   export OPENAI_API_KEY=sk-...                                       # bbcqa / tiebe need a key (from the env)
#   sbatch examples/slurm/inference.slurm                             #   (sbatch propagates the submitting env)
# On an account/partition cluster (e.g. nano5), set these once — Slurm reads them, no file edit:
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>

#SBATCH --job-name=gadra-eval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err

set -euo pipefail

# --- Paths: EDIT to your shared filesystem (must be visible from BOTH login and compute nodes) ---
REPO="${GADRA_REPO:-/path/to/shared/GaDRA}"
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

# Config-driven eval — edit examples/config/inference.yaml (task / adapter / eval_file / documents_file),
# or override on the CLI, e.g.:  ... -m examples.inference --config examples/config/inference.yaml --override task=bbcqa
uv run --no-sync --offline python -m examples.inference \
    --config examples/config/inference.yaml
