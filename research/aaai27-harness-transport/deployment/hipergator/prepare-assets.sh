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
TRACEH_PYTHON="${TRACEH_PYTHON:-$TRACEH_ROOT/envs/traceh/bin/python}"
TRACEH_MODEL_REPO="${TRACEH_MODEL_REPO:-Qwen/Qwen3-4B}"
TRACEH_MODEL_NAME="${TRACEH_MODEL_NAME:-Qwen3-4B}"
TRACEH_MODEL_REVISION="${TRACEH_MODEL_REVISION:-main}"
MODEL_DIR="$TRACEH_ROOT/models/$TRACEH_MODEL_NAME"
DATA_DIR="$TRACEH_ROOT/data/alfworld"
ASSET_MANIFEST_DIR="$TRACEH_ROOT/manifests/assets"

case "$TRACEH_ROOT" in
  /blue/*) ;;
  *) echo "ERROR: TRACEH_ROOT must be under /blue: $TRACEH_ROOT" >&2; exit 2 ;;
esac

test -x "$TRACEH_PYTHON"
test -f "$TRACEH_PROJECT_ROOT/deployment/hipergator/make_asset_manifest.py"
HF_BIN="$(dirname "$TRACEH_PYTHON")/hf"
ALFWORLD_DOWNLOAD="$(dirname "$TRACEH_PYTHON")/alfworld-download"
test -x "$HF_BIN"
test -x "$ALFWORLD_DOWNLOAD"
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$ASSET_MANIFEST_DIR"

if [[ "$TRACEH_MODEL_REVISION" == "main" ]]; then
  echo "WARNING: model revision is main; acceptable for H1 smoke only, not formal final" >&2
fi

"$HF_BIN" download "$TRACEH_MODEL_REPO" \
  --revision "$TRACEH_MODEL_REVISION" \
  --local-dir "$MODEL_DIR" \
  --cache-dir "$TRACEH_ROOT/.cache/huggingface" \
  --max-workers 8

if [[ ! -d "$DATA_DIR/json_2.1.1/valid_seen" ]]; then
  "$ALFWORLD_DOWNLOAD" --data-dir "$DATA_DIR"
fi

MODEL_MANIFEST="$ASSET_MANIFEST_DIR/$TRACEH_MODEL_NAME.manifest.json"
DATA_MANIFEST="$ASSET_MANIFEST_DIR/alfworld-0.4.2.manifest.json"

"$TRACEH_PYTHON" "$TRACEH_PROJECT_ROOT/deployment/hipergator/make_asset_manifest.py" \
  --root "$MODEL_DIR" \
  --asset-id "$TRACEH_MODEL_NAME" \
  --source "https://huggingface.co/$TRACEH_MODEL_REPO" \
  --revision "$TRACEH_MODEL_REVISION" \
  --output "$MODEL_MANIFEST"

"$TRACEH_PYTHON" "$TRACEH_PROJECT_ROOT/deployment/hipergator/make_asset_manifest.py" \
  --root "$DATA_DIR" \
  --asset-id "ALFWorld-0.4.2" \
  --source "alfworld-download==0.4.2" \
  --revision "json_2.1.1" \
  --output "$DATA_MANIFEST"

printf 'TRACEH_MODEL_PATH=%s\n' "$MODEL_DIR"
printf 'TRACEH_CHECKPOINT_MANIFEST=%s\n' "$MODEL_MANIFEST"
printf 'TRACEH_CHECKPOINT_HASH=%s\n' "$(sha256sum "$MODEL_MANIFEST" | cut -d' ' -f1)"
printf 'TRACEH_ALFWORLD_DATA=%s\n' "$DATA_DIR"
printf 'TRACEH_ALFWORLD_MANIFEST=%s\n' "$DATA_MANIFEST"
printf 'TRACEH_ALFWORLD_HASH=%s\n' "$(sha256sum "$DATA_MANIFEST" | cut -d' ' -f1)"
