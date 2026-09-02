#!/usr/bin/env bash
set -euo pipefail

cd /home/victorlin/Documents/anvil-embodied-ai

export PATH="$HOME/.local/bin:$PATH"
source /srv/shared/env/huggingface.sh
export UV_CACHE_DIR=/srv/shared/cache/uv
export PIP_CACHE_DIR=/srv/shared/cache/pip

uv run anvil-trainer \
  --dataset.root=/home/victorlin/Documents/openarm_dev/anvil/datasets/macp_20260813__pass \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --task-description="Use the left hand to pick up the plastic bottle from the table, hand it over to the right hand, then place it back on the table." \
  --job_name=smolvla_smoke_test \
  --output_dir=/srv/shared/model_zoo/anvil/smoke_tests/smolvla_smoke_test \
  --steps=50 \
  --batch_size=1 \
  --wandb.enable=false
