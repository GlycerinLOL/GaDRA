#!/bin/bash
# GaDRA multi-GPU continual pre-training under SLURM, managed with uv (no conda).
#
# Single-node, N-GPU: `accelerate launch` + DeepSpeed ZeRO-2 (one process per GPU). The gradient-accumulation
# steps are auto-computed so the paper's global batch size (128) is held no matter how many GPUs you request.
#
# TWO-PHASE workflow (compute nodes are typically offline):
#   1) ON THE LOGIN NODE (has internet), ONCE, before submitting — populate the shared uv cache + .venv:
#        export GADRA_REPO=/path/to/shared/GaDRA   # checkout on a filesystem both node types can see
#        export UV_CACHE_DIR="$GADRA_REPO/.uv-cache"
#        export UV_PYTHON_INSTALL_DIR="$GADRA_REPO/.uv-python"
#        cd "$GADRA_REPO" && uv sync --group gpu --locked
#      (downloads cu124 torch + the prebuilt flash-attn wheel + deepspeed into the cache; commits nothing.)
#   2) Submit:
#        sbatch examples/slurm/train.slurm                      # uses the GPUs allocated below (default 2)
#        sbatch --gpus-per-node=4 examples/slurm/train.slurm    # 4 GPUs — grad-accum auto-adjusts to GBS 128
#
# On an account/partition cluster (e.g. nano5), set these once — Slurm reads them, no file edit:
#   export SBATCH_ACCOUNT=<account> SBATCH_PARTITION=<partition>
#
# Knobs (env vars; paper defaults): N_GPUS (auto-detected), MBS=16 (per-device batch), GBS=128 (global batch).
# Multi-NODE is out of scope here (single node); ask if you need cross-node rendezvous.

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

# --- GPU count: prefer Slurm's per-node allocation, else count visible GPUs, else 1 ---
N_GPUS="${N_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]] || [ "${N_GPUS}" -lt 1 ]; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
fi
[ "${N_GPUS:-0}" -lt 1 ] && N_GPUS=1

# --- Hold the paper's global batch size: GBS = per_device_bs * grad_accum * N_GPUS = 128 ---
MBS="${MBS:-16}"                       # per_device_train_batch_size (paper)
GBS="${GBS:-128}"                      # target global batch size (paper invariant)
GA=$(( GBS / (MBS * N_GPUS) )); [ "${GA}" -lt 1 ] && GA=1
EFF_GBS=$(( MBS * GA * N_GPUS ))
echo "GPUs=${N_GPUS}  per_device_bs=${MBS}  grad_accum=${GA}  ->  global_batch=${EFF_GBS} (target ${GBS})"
[ "${EFF_GBS}" -ne "${GBS}" ] && \
    echo "WARNING: effective global batch ${EFF_GBS} != target ${GBS}; adjust GPU count / MBS / GBS." >&2

# --- Launch: accelerate + DeepSpeed ZeRO-2, one process per GPU; grad-accum keeps GBS fixed ---
uv run --no-sync --offline accelerate launch \
    --num_machines "${SLURM_JOB_NUM_NODES:-1}" \
    --num_processes "${N_GPUS}" \
    --config_file examples/config/deepspeed_zero2.yaml \
    -m examples.train --config examples/config/train.yaml \
    --override per_device_train_batch_size="${MBS}" \
    --override gradient_accumulation_steps="${GA}"

# Single-GPU variant (no accelerate/deepspeed; train.yaml already encodes GBS 128 = 16 x 8):
#   uv run --no-sync --offline python -m examples.train --config examples/config/train.yaml
