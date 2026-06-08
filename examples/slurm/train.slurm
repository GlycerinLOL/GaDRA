#!/bin/bash
# GaDRA multi-GPU continual pre-training (accelerate + DeepSpeed ZeRO-2) under SLURM (uv).
# See examples/slurm/README.md for setup, knobs, and submission examples.

#SBATCH --job-name=gadra-cpt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
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
export PYTHONUNBUFFERED=1

# --- Isolate from conda ---
unset CONDA_PREFIX CONDA_DEFAULT_ENV 2>/dev/null || true

cd "$REPO"
mkdir -p slurm-logs

# --- GPU count ---
N_GPUS="${N_GPUS:-${SLURM_GPUS_ON_NODE:-}}"
if ! [[ "${N_GPUS}" =~ ^[0-9]+$ ]] || [ "${N_GPUS}" -lt 1 ]; then
    N_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)
fi
[ "${N_GPUS:-0}" -lt 1 ] && N_GPUS=1

# --- Global batch size (GBS = MBS * grad_accum * N_GPUS) ---
MBS="${MBS:-16}"                       # per_device_train_batch_size
GBS="${GBS:-128}"                      # target global batch size
GA=$(( GBS / (MBS * N_GPUS) )); [ "${GA}" -lt 1 ] && GA=1
EFF_GBS=$(( MBS * GA * N_GPUS ))
echo "GPUs=${N_GPUS}  per_device_bs=${MBS}  grad_accum=${GA}  ->  global_batch=${EFF_GBS} (target ${GBS})"
[ "${EFF_GBS}" -ne "${GBS}" ] && \
    echo "WARNING: effective global batch ${EFF_GBS} != target ${GBS}; adjust GPU count / MBS / GBS." >&2

# --- Method + config ---
CONFIG="${CONFIG:-examples/config/train.yaml}"
METHOD_OVERRIDE=()
if [ -n "${METHOD:-}" ]; then METHOD_OVERRIDE=(--override "method=${METHOD}"); echo "method: ${METHOD}"; fi
echo "config: ${CONFIG}"

# --- Output dir: save/<DATASET>/<DATE>/CPT/<exp_name> ---
# The method-related last component (exp_name) defaults from the run-config's exp_name: field;
# set EXP_NAME to override it without editing the config.
DATASET="${DATASET:-BBC_news}"
DATE="${DATE:-$(date +%Y%m%d)}"
if [ -z "${EXP_NAME:-}" ]; then
    EXP_NAME=$(uv run --no-sync --offline python -c \
        "import sys; from examples import load_run_config; print(load_run_config(sys.argv[1]).get('exp_name') or '')" \
        "${CONFIG}")
fi
[ -n "${DATASET}" ] && [ -n "${EXP_NAME}" ] && [ -n "${DATE}" ] || { echo "ERROR: DATASET/DATE required and exp_name must be set in ${CONFIG} (or via EXP_NAME)" >&2; exit 2; }
SAVE="save/${DATASET}/${DATE}/CPT/${EXP_NAME}"
rm -rf "${SAVE}"          # same dataset+date+exp_name re-run overwrites
mkdir -p "${SAVE}"
echo "output_dir: ${SAVE}"

# --- Launch ---
uv run --no-sync --offline accelerate launch \
    --num_machines "${SLURM_JOB_NUM_NODES:-1}" \
    --num_processes "${N_GPUS}" \
    --config_file examples/config/deepspeed_zero2.yaml \
    -m examples.train --config "${CONFIG}" \
    "${METHOD_OVERRIDE[@]}" \
    --override per_device_train_batch_size="${MBS}" \
    --override gradient_accumulation_steps="${GA}" \
    --override output_dir="${SAVE}" 2>&1 | tee "${SAVE}/log.txt"

# Single-GPU variant (no accelerate/deepspeed; train.yaml already encodes GBS 128 = 16 x 8):
#   uv run --no-sync --offline python -m examples.train --config "${CONFIG}" "${METHOD_OVERRIDE[@]}" 2>&1 | tee "${SAVE}/log.txt"
