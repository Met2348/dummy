#!/usr/bin/env bash
#SBATCH --job-name=traceh-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=traceh-preflight-%j.out
#SBATCH --export=ALL

set -euo pipefail

PROJECT_ROOT="${TRACEH_PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${TRACEH_PYTHON:-python}"

case "$PROJECT_ROOT" in
  /blue/*) ;;
  *)
    echo "ERROR: TRACEH_PROJECT_ROOT must be under /blue, got: $PROJECT_ROOT" >&2
    exit 2
    ;;
esac

test -d "$PROJECT_ROOT"
command -v nvidia-smi >/dev/null
command -v "$PYTHON_BIN" >/dev/null

echo "job_id=$SLURM_JOB_ID"
echo "job_account=${SLURM_JOB_ACCOUNT:-unknown}"
echo "node=$(hostname)"
echo "project_root=$PROJECT_ROOT"
echo "slurm_tmpdir=${SLURM_TMPDIR:-unset}"
echo "python=$PYTHON_BIN"
date --iso-8601=seconds
module list 2>&1 || true
nvidia-smi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import platform

import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the selected runtime")

print(f"gpu_name={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
for package in ("transformers", "vllm", "flash_attn", "ot", "gymnasium", "alfworld"):
    print(f"importable_{package}={importlib.util.find_spec(package) is not None}")

x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
y = x @ x
torch.cuda.synchronize()
print(f"matmul_checksum={float(y[0, 0])}")
print(f"peak_allocated_bytes={torch.cuda.max_memory_allocated()}")
PY

date --iso-8601=seconds

