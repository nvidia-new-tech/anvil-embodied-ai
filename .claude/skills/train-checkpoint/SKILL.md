---
name: train-checkpoint
description: Use when training a SmolVLA or pi0.5 policy on a shared LeRobot dataset (anvil-trainer), resuming an interrupted training run, or promoting a real-robot-validated checkpoint into /srv/shared/model_zoo.
---

# Train a checkpoint on a shared dataset

## Overview

Dataset → `anvil-trainer` with a validated hyperparameter recipe → checkpoint →
(optional) promote to shared model_zoo once real-robot-tested. `anvil-trainer` has
no early stopping or best-checkpoint tracking — pick checkpoints by real-robot
result, not val loss (val loss has repeatedly picked the wrong checkpoint in this
repo's history: a lower-loss earlier step performed worse on the physical robot
than a later, higher-loss one).

## Step 1 — Pick the recipe

Default to **full data, no split** unless you specifically need val/test:
`--split-ratio=1,0,0`. This is not the trainer's own default — omitting it
silently reintroduces an 8:1:1 split.

**SmolVLA:**
```bash
uv run anvil-trainer \
  --dataset.root=/srv/shared/datasets/anvil/lerobot/<name> \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --split-ratio=1,0,0 \
  --batch_size=64 --num_workers=8 \
  --steps=5000 --save_freq=1000 --log_freq=100 \
  --job_name=<policy>-<dataset> \
  --task-description="<task string>" \
  --wandb.enable=false
```

**pi0.5:**
```bash
uv run anvil-trainer \
  --dataset.root=/srv/shared/datasets/anvil/lerobot/<name> \
  --policy.type=pi05 \
  --policy.pretrained_path=model_zoo/pi05_base_local \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=false \
  --policy.freeze_vision_encoder=false \
  --policy.gradient_checkpointing=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.optimizer_lr=5e-5 \
  --split-ratio=1,0,0 \
  --batch_size=16 --num_workers=8 \
  --steps=5000 --save_freq=1000 --log_freq=100 \
  --job_name=<policy>-<dataset> \
  --task-description="<task string>" \
  --wandb.enable=false
```

`train_expert_only=false` + `gradient_checkpointing=true` at `batch_size=16` is the
validated combo on a 96GB card (3 cameras). `train_expert_only=true` fits more
memory headroom and a bigger batch, but the official LeRobot docs state this
configuration explicitly as **"less memory, reduced success rate"** — don't reach
for it as a default just because it's cheaper; it under-trains the model.
`QUANTILES` normalization (pi0.5's own default) isn't wrong by itself, but
`MEAN_STD` is what's validated working here and matches SmolVLA, so use it unless
you have a specific reason not to.

**Real hardware requirement:** run `scripts/preflight_checkpoint.py <ckpt> --config <yaml>`
before ever trusting a new checkpoint on a robot — camera/state-width/action-slice
mismatches between the checkpoint and inference config fail *silently* (blank
image, policy stares at grey frame; or an 8-DOF checkpoint gets fed a 16-DOF
vector and reads left-arm values into right-arm slots). This tool exists
specifically because those failures produce plausible-looking motion, not an
error message.

## Step 2 — Launch and monitor

```bash
mkdir -p logs
export HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
nohup uv run anvil-trainer <args> >> logs/<job_name>.log 2>&1 &
disown
```

Wait for completion by polling the **process**, not a log-text match:
```bash
while ps -p <pid> > /dev/null 2>&1; do sleep 20; done
```
Don't `grep -q "End of training" logfile` on a log file that's being *appended to*
across resumes — a prior run's completion line is already in the file and matches
immediately, even though the current segment is still running.

## Step 3 — Resume after interruption

The training session/sandbox can die mid-run for reasons outside the job itself
(host suspend, session teardown, OOM elsewhere) — this doesn't mean the run
crashed. Checkpoints saved at `save_freq` make this cheap to recover from:

```bash
uv run anvil-trainer \
  --resume=model_zoo/<dataset>/<job_name> \
  --steps=<original target> \
  --split-ratio=1,0,0 \
  --note-append="resumed after interruption at step N"
```

**`--split-ratio` and other CLI overrides are NOT auto-restored from the resumed
checkpoint's config** — re-pass every non-default flag on every resume, or the
job silently falls back to defaults (e.g. re-introduces the 8:1:1 split you
disabled originally).

**Duplicate-process trap:** if a "did it die?" check gives a false positive (e.g.
a flaky `pgrep -f` pattern) and you launch a second resume from the same
checkpoint while the first is still alive, you get two processes training
concurrently against the same output directory. Before resuming, confirm the
original PID is actually gone (`ps -p <pid>`), not just that some pattern-matched
`pgrep` returned empty.

## Step 4 — Export off the training machine (optional)

Copy `pretrained_model/` only — never `training_state/` (optimizer/scheduler
state, large, not needed for inference):

```bash
rsync -a --info=progress2 \
  <ckpt_dir>/pretrained_model/ /media/<drive>/<name>/ \
  && sync
```

`sync` matters here too — `rsync` returns once data is buffered, not once it's
physically on the (often much slower) USB device. Don't eject before `sync`
completes.

## Step 5 — Promote to shared model_zoo (after real-robot validation only)

Do **not** promote a checkpoint just because training finished cleanly. Promote
only once it's been tested on the physical robot and the result is worth keeping
around for others:

```bash
mkdir -p /srv/shared/model_zoo/anvil/<dataset>/<policy>_<dataset>_downstream/checkpoints/<step>
rsync -a <ckpt>/pretrained_model/ \
  /srv/shared/model_zoo/anvil/<dataset>/<policy>_<dataset>_downstream/checkpoints/<step>/pretrained_model/
sync
```

Write `model_card.yaml` alongside it: `name`, `policy_type`, `base_model`,
`dataset`, `dataset_path`, `task_description`, `steps`, `batch_size`,
`normalization`, `created_by`, `created_at`, `checkpoint_path`, `status`
(`candidate` until confirmed reliable, `production` once it is), and a `notes:`
block with the actual real-robot result (failure modes observed, comparison
against other checkpoints if a benchmark exists). Everything else — exploratory
runs, checkpoints that haven't been tested yet — stays in the local `model_zoo/`,
not shared.

## Common Mistakes

| Symptom | Cause | Fix |
|---|---|---|
| Training silently split 8:1:1 after a resume | `--split-ratio` isn't inherited from the checkpoint on `--resume` | Re-pass every CLI flag on every resume |
| Two processes training against the same checkpoint dir | A stale/flaky process-liveness check false-positived "died", triggering a duplicate resume | Confirm the exact PID is gone before resuming, not just a pattern match |
| Job looks "done" immediately after a resume, but it hasn't trained any new steps | `grep "End of training"` matched a stale line from a prior completed segment in the same (appended) log file | Poll the process itself, not log text |
| pi0.5 real-robot performance is worse than expected despite a healthy loss curve | `train_expert_only=true` was used for memory headroom | Use `train_expert_only=false` + `gradient_checkpointing=true` instead; adjust batch size down if needed |
| A checkpoint that trains cleanly performs badly on the robot | Not necessarily a training bug — check the *inference config* first (camera set mismatch, wrong arm mapping) before assuming the model is bad | Run `preflight_checkpoint.py` and `offline_inference_test.py` before blaming training |
