#!/usr/bin/env bash
set -euo pipefail

cd /home/victorlin/Documents/anvil-embodied-ai

export PATH="$HOME/.local/bin:$PATH"
source /srv/shared/env/huggingface.sh
export UV_CACHE_DIR=/srv/shared/cache/uv
export PIP_CACHE_DIR=/srv/shared/cache/pip

DATASET_ROOT="${DATASET_ROOT:-/home/victorlin/Documents/openarm_dev/anvil/datasets/macp_20260813__pass}"
OUTPUT_DIR="${OUTPUT_DIR:-/srv/shared/model_zoo/anvil/smoke_tests/pi05_smoke_test}"
JOB_NAME="${JOB_NAME:-pi05_smoke_test}"
STEPS="${STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Use the left hand to pick up the plastic bottle from the table, hand it over to the right hand, then place it back on the table.}"

uv run anvil-trainer \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=false \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --task-description="$TASK_DESCRIPTION" \
  --job_name="$JOB_NAME" \
  --output_dir="$OUTPUT_DIR" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers=0 \
  --wandb.enable=false
