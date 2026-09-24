#!/usr/bin/env bash
# Remove training_state/ from every checkpoint except the one `last` points to.
# Usage:
#   scripts/prune_training_state.sh [ROOT]        # dry-run (default ROOT=model_zoo)
#   scripts/prune_training_state.sh [ROOT] --apply
set -euo pipefail

ROOT="${1:-model_zoo}"
APPLY=0
for a in "$@"; do [[ "$a" == "--apply" ]] && APPLY=1; done

[[ -d "$ROOT" ]] || { echo "ROOT not found: $ROOT" >&2; exit 1; }

total_bytes=0
n_del=0

while IFS= read -r -d '' ckpt_root; do
  last_link="$ckpt_root/last"
  keep=""
  if [[ -e "$last_link" ]]; then
    keep="$(cd "$ckpt_root" && readlink -f last)"
  else
    echo "[warn] no 'last' in $ckpt_root -- skipping whole dir (nothing deleted)"
    continue
  fi

  while IFS= read -r -d '' ts; do
    step_dir="$(cd "$(dirname "$ts")" && pwd -P)"
    if [[ "$step_dir" == "$keep" ]]; then
      echo "[keep] $ts"
      continue
    fi
    sz=$(du -sb "$ts" | cut -f1)
    total_bytes=$((total_bytes + sz))
    n_del=$((n_del + 1))
    if [[ $APPLY -eq 1 ]]; then
      rm -rf -- "$ts"
      echo "[del ] $ts ($(numfmt --to=iec "$sz"))"
    else
      echo "[dry ] $ts ($(numfmt --to=iec "$sz"))"
    fi
  done < <(find "$ckpt_root" -mindepth 2 -maxdepth 2 -type d -name training_state -print0)
done < <(find "$ROOT" -type d -name checkpoints -print0)

echo "----"
echo "$( ((APPLY)) && echo 'Deleted' || echo 'Would delete') $n_del training_state dir(s), $(numfmt --to=iec "$total_bytes")"
((APPLY)) || echo "Dry-run. Re-run with --apply to actually delete."
