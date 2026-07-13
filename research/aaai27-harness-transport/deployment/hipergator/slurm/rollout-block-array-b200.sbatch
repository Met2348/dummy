#!/usr/bin/env bash
#SBATCH --job-name=traceh-b200
#SBATCH --nodes=1
#SBATCH --partition=hpg-b200
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=12:00:00
#SBATCH --output=traceh-b200-%A_%a.out
#SBATCH --export=ALL

set -euo pipefail

for name in TRACEH_PROJECT_ROOT TRACEH_BLOCK_MANIFEST TRACEH_BLOCK_RUNNER TRACEH_OUTPUT_ROOT; do
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: required environment variable is unset: $name" >&2
    exit 2
  fi
done

case "$TRACEH_PROJECT_ROOT" in
  /blue/*) ;;
  *) echo "ERROR: TRACEH_PROJECT_ROOT must be under /blue" >&2; exit 2 ;;
esac
case "$TRACEH_OUTPUT_ROOT" in
  /blue/*) ;;
  *) echo "ERROR: TRACEH_OUTPUT_ROOT must be under /blue" >&2; exit 2 ;;
esac

test -f "$TRACEH_BLOCK_MANIFEST"
test -x "$TRACEH_BLOCK_RUNNER"

BLOCK_INDEX="$SLURM_ARRAY_TASK_ID"
ATTEMPT_ID="${SLURM_ARRAY_JOB_ID}_${BLOCK_INDEX}_${SLURM_JOB_ID}"
OUTPUT_DIR="$TRACEH_OUTPUT_ROOT/$ATTEMPT_ID"
mkdir -p "$OUTPUT_DIR"

export TRACEH_GPU_CLASS=b200
export TRACEH_RUN_OUTPUT="$OUTPUT_DIR"
export HF_HOME="${SLURM_TMPDIR:?}/hf-cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

if [[ -n "${TRACEH_MODEL_SOURCE:-}" ]]; then
  test -d "$TRACEH_MODEL_SOURCE"
  mkdir -p "$SLURM_TMPDIR/model"
  rsync -a "$TRACEH_MODEL_SOURCE/" "$SLURM_TMPDIR/model/"
  export TRACEH_MODEL_PATH="$SLURM_TMPDIR/model"
fi

nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
date --iso-8601=seconds
srun "$TRACEH_BLOCK_RUNNER" \
  --manifest "$TRACEH_BLOCK_MANIFEST" \
  --block-index "$BLOCK_INDEX" \
  --output-dir "$OUTPUT_DIR"
date --iso-8601=seconds

