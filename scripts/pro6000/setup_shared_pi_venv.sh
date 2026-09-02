#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a per-user .venv while reusing shared model/package caches.
# Run this from any user clone of anvil-embodied-ai.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

SHARED_ROOT="${SHARED_ROOT:-/srv/shared}"
HF_HOME="${HF_HOME:-$SHARED_ROOT/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$SHARED_ROOT/cache/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SHARED_ROOT/cache/pip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---all-packages --extra smolvla --extra pi}"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "ERROR: uv was not found. Install uv for this user first, then rerun this script." >&2
  echo "Expected one of: uv in PATH, or $HOME/.local/bin/uv" >&2
  exit 1
fi

require_dir() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo "ERROR: missing directory: $dir" >&2
    exit 1
  fi
}

require_readable_dir() {
  local dir="$1"
  require_dir "$dir"
  if [ ! -r "$dir" ] || [ ! -x "$dir" ]; then
    echo "ERROR: cannot read or traverse $dir" >&2
    echo "Check that this user is in the shared group, for example: marvinteam." >&2
    exit 1
  fi
}

require_writable_dir() {
  local dir="$1"
  require_readable_dir "$dir"
  if [ ! -w "$dir" ]; then
    echo "ERROR: cannot write to $dir" >&2
    echo "Package cache needs group write permission for shared reuse." >&2
    exit 1
  fi
}

require_readable_dir "$HF_HOME"
require_readable_dir "$HF_HUB_CACHE"
require_writable_dir "$UV_CACHE_DIR"
require_writable_dir "$PIP_CACHE_DIR"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME
export HF_HUB_CACHE
export TRANSFORMERS_CACHE
export UV_CACHE_DIR
export PIP_CACHE_DIR
# Avoid hardlink warnings or cross-user ownership surprises when cache and venv differ.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

if [ ! -d .venv ]; then
  "$UV_BIN" venv .venv --python "$PYTHON_VERSION"
else
  echo "Using existing .venv"
fi

"$UV_BIN" sync $UV_SYNC_ARGS

cat <<REPORT

Pi0/Pi0.5 environment is ready.
Repo:            $REPO_ROOT
Virtualenv:      $REPO_ROOT/.venv
Hugging Face:    $HF_HOME
HF hub cache:    $HF_HUB_CACHE
uv cache:        $UV_CACHE_DIR
pip cache:       $PIP_CACHE_DIR

Activate with:
  cd "$REPO_ROOT"
  source .venv/bin/activate
REPORT
