#!/bin/bash
# GaDRA multi-GPU continual pre-training under SLURM, managed with uv (no conda).
#
# TWO-PHASE workflow (compute nodes are typically offline):
#   1) ON THE LOGIN NODE (has internet), ONCE, before submitting — populate the shared uv cache + .venv:
#        export GADRA_REPO=/path/to/shared/GaDRA   # your checkout, on a filesystem both node types can see
#        export UV_CACHE_DIR="$GADRA_REPO/.uv-cache"
#        export UV_PYTHON_INSTALL_DIR="$GADRA_REPO/.uv-python"
#        cd "$GADRA_REPO" && uv sync --group gpu --locked
#      (downloads cu124 torch + the prebuilt flash-attn wheel + deepspeed into the cache; commits nothing.)
#   2) Submit this script:  sbatch examples/slurm/train.slurm
#
# On an account/partition cluster (e.g. nano5), set these once — Slurm reads them, no file edit:
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
#
# Everything below runs on an OFFLINE compute node and reuses the login-node env on the shared filesystem.

#SBATCH --job-name=gadra-cpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
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

# --- Isolate from conda (the account keeps conda for the parent research repo; do NOT let it bleed in) ---
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true

cd "$REPO"
mkdir -p slurm-logs

# Multi-GPU = the paper's workflow: accelerate launch + DeepSpeed ZeRO-2, one process per GPU.
N_GPUS="${SLURM_GPUS_PER_NODE:-2}"

uv run --no-sync --offline accelerate launch \
    --num_machines "${SLURM_JOB_NUM_NODES:-1}" \
    --num_processes "$N_GPUS" \
    --config_file examples/config/deepspeed_zero2.yaml \
    -m examples.train --config examples/config/train.yaml

# Single-GPU variant (drop accelerate/deepspeed):
#   uv run --no-sync --offline python -m examples.train --config examples/config/train.yaml
