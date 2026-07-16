#!/usr/bin/env bash
set -euo pipefail

: "${HPG_ACCOUNT:?Set HPG_ACCOUNT to the HiPerGator group/account name}"

TRACEH_ROOT="${TRACEH_ROOT:-/blue/$HPG_ACCOUNT/$USER/trace-h}"
if [[ -z "${TRACEH_PROJECT_ROOT:-}" ]]; then
  if [[ -f "$TRACEH_ROOT/repo/traceh/experiments/pyproject.toml" ]]; then
    TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/traceh"
  else
    TRACEH_PROJECT_ROOT="$TRACEH_ROOT/repo/dummy/research/aaai27-harness-transport"
  fi
fi
TRACEH_ENV="${TRACEH_ENV:-$TRACEH_ROOT/envs/traceh}"
BOOTSTRAP_PYTHON="${TRACEH_BOOTSTRAP_PYTHON:-python3}"
UV_TOOL_ENV="$TRACEH_ROOT/tools/uv"

case "$TRACEH_ROOT" in
  /blue/*) ;;
  *) echo "ERROR: TRACEH_ROOT must be under /blue: $TRACEH_ROOT" >&2; exit 2 ;;
esac

test -f "$TRACEH_PROJECT_ROOT/experiments/pyproject.toml"
test -f "$TRACEH_PROJECT_ROOT/experiments/uv.lock"
command -v "$BOOTSTRAP_PYTHON" >/dev/null

mkdir -p \
  "$TRACEH_ROOT"/{envs,tools,models,data,manifests,artifacts,runs,logs} \
  "$TRACEH_ROOT/.cache/uv" \
  "$TRACEH_ROOT/.cache/huggingface"

export UV_CACHE_DIR="$TRACEH_ROOT/.cache/uv"
export HF_HOME="$TRACEH_ROOT/.cache/huggingface"
export UV_PROJECT_ENVIRONMENT="$TRACEH_ENV"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  if [[ ! -x "$UV_TOOL_ENV/bin/uv" ]]; then
    "$BOOTSTRAP_PYTHON" -m venv "$UV_TOOL_ENV"
    "$UV_TOOL_ENV/bin/python" -m pip install --upgrade pip
    "$UV_TOOL_ENV/bin/python" -m pip install 'uv>=0.7,<1'
  fi
  UV_BIN="$UV_TOOL_ENV/bin/uv"
fi

"$UV_BIN" sync \
  --project "$TRACEH_PROJECT_ROOT/experiments" \
  --frozen \
  --extra agent \
  --extra inference \
  --group dev

"$TRACEH_ENV/bin/python" - <<'PY'
import importlib.util
import platform

print(f"python={platform.python_version()}")
for name in ("torch", "transformers", "accelerate", "bitsandbytes", "alfworld", "ot"):
    print(f"importable_{name}={importlib.util.find_spec(name) is not None}")
PY

printf 'TRACEH_PROJECT_ROOT=%s\n' "$TRACEH_PROJECT_ROOT"
printf 'TRACEH_PYTHON=%s\n' "$TRACEH_ENV/bin/python"
printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
printf 'HF_HOME=%s\n' "$HF_HOME"
