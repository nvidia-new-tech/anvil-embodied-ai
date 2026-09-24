---
name: mcap-to-dataset
description: Use when a new MCAP recording session needs converting into a shared LeRobot dataset — e.g. data just copied from a USB drive, a new robot rig, or any raw session under /srv/shared/datasets/Anvil_raw_data that has no matching entry under /srv/shared/datasets/anvil/lerobot yet.
---

# MCAP → shared LeRobot dataset

## Overview

Fixed pipeline: raw MCAP session → quality scan → mcap-convert → dataset-valid →
dataset_card.yaml, landing in `/srv/shared/datasets/anvil/lerobot/<task>_<date>_<variant>/`.
Every step here has caused a real bug in this repo's history — read Common Mistakes
before running, not after.

**Ask, don't assume.** This pipeline has several points where a wrong silent
default has corrupted data or broken training in the past. At each "Checkpoint"
below, you MUST actually stop and ask a real question (e.g. the
AskUserQuestion tool, or a plain question in chat if that tool isn't
available) and wait for the answer before continuing to the next step. A
checkpoint is not satisfied by proceeding and then *mentioning* what you did
in a final summary — by the time you report, the config is already written or
the conversion already ran, and the user is confirming your work, not making
the decision.

This applies even when auto-mode is active, even when the task looks
pre-scoped or low-stakes (a "just a test/dummy dataset" framing is exactly the
case where a bad default goes unnoticed), and even when every value can be
correctly derived from the raw MCAP data — deriving something correctly and
getting sign-off on it before acting are different things. If you truly
cannot ask (no user available at all, e.g. fully unattended batch job), say so
explicitly and treat every checkpoint's default as unconfirmed in your report,
rather than presenting the result as reviewed.

## Step 1 — Copy raw data to shared storage

```bash
rsync -a --info=progress2 /media/<user>/<drive>/<session>/ \
  /srv/shared/datasets/Anvil_raw_data/<task>_<date>_<variant>/
sync
```

`sync` matters: `rsync` returns as soon as data hits the page cache, not when it's
physically on disk. Don't unplug/eject before `sync` finishes.

## Step 2 — Inspect the raw MCAP to determine rig shape

Decode a handful of messages from one episode (don't need mcap-valid for this, just
`mcap_ros2.decoder.DecoderFactory` on one file) to determine:

- **Cameras present and native resolution** — decide `image_resolution` as an exact
  integer downscale (÷2, ÷4, ...) of native, or unchanged if already small. Never
  pick a resolution that isn't an exact divisor — it letterboxes.
- **Arm(s) actually commanded** — check `/joint_states` for which `follower_l_*` /
  `follower_r_*` prefixes exist, AND check whether a corresponding command topic
  (`/follower_{arm}_forward_position_controller/commands`) has real messages. A
  topic existing with the right camera/joint *name* but zero real commands (arm
  physically present but never driven) is common — see the single-arm decision
  below.
- **Action message type**: `std_msgs/Float64MultiArray` (joint-space, the default)
  or `anvil_msgs/CommandedEEPose` (EE-space — position + quaternion + gripper).
  Check the actual topic list; don't assume from the task name.
- **fps**: cameras almost always run ~60Hz; use `--fps 30` (exact ÷2, matches
  every session in this repo's history) unless you have a specific reason not to.

**Checkpoint — ask before writing the config:** use AskUserQuestion with these
selections (don't fold them into one free-text question):

- **Cameras** (multiSelect): one option per camera topic found, pre-described
  with its native resolution and proposed downscale. Let the user pick which
  to include, don't default to "all."
- **Arm(s)** (multiSelect if bimanual rig, single-select if only one arm has
  real data): one option per arm found, noting whether it's genuinely driven
  or only recorded-idle. If only one arm has real commands, still confirm
  single-arm vs hold-position-fallback rather than assuming.
- **Action space** (single-select): "Joint-space (Float64MultiArray)" vs
  "EE-space (CommandedEEPose)" — state which one the raw topics actually show,
  but let the user confirm rather than silently trusting the detection.

fps does not need a question — use the proposed `--fps 30` (or native fps if
already ≤30) automatically and just state it in the summary.

Task description: give a suggested wording as one of the options, plus let the
user type their own via free text (AskUserQuestion's "Other" always allows
this) — don't finalize wording without either an explicit pick or custom
input.

## Step 3 — Quality scan

```bash
uv run mcap-valid --input /srv/shared/datasets/Anvil_raw_data/<name>
```

Read the report before converting, not after — most flags across every session
this repo has processed are **false positives**, and mcap-convert's defaults skip
`critical` episodes silently unless overridden:

- A camera or command topic the config will never reference (e.g. `wrist_l`,
  `waist` on a rig that never actually used it) shows "zero messages" as
  `critical` even though nothing is actually wrong. Safe to override with
  `--include-flagged critical` once you've confirmed the flagged topic isn't
  one your config uses.
- A genuinely corrupted episode (`RecordLengthLimitExceeded`, footer truncated —
  recorder/disk killed mid-write) needs `--skip-episode-idx`. **The index is the
  1-based *position* in sorted file order, not the folder number** — if folder
  numbering has gaps (common), find the real position first:
  ```bash
  ls <raw_dir> | grep -E "^[0-9]{4}$" | sort | nl | grep <folder_number>
  ```

## Step 4 — Write or pick the mcap_converter config

Copy the closest existing config under `configs/mcap_converter/openarm_single_quest_*.yaml`
as a template. Key decision:

```
Is the arm/camera genuinely unused (never driven / zero real data),
not just recorded-but-idle?
  │
  ├─ yes → don't declare it in `arms:` / `camera_topics:` at all.
  │        parse_joint_name silently drops unmapped arm prefixes — no
  │        hold-position padding, no phantom near-zero-variance dimension
  │        that later breaks QUANTILES normalization at training time.
  │
  └─ no (arm present, genuinely idle sometimes, e.g. one-armed task on a
        bimanual rig with both arms wired) → keep both arms declared;
        the observation-as-action fallback fills the idle arm's action
        from its own observation (hold position).
```

For EE-space actions, the topic config needs `msg_type: "CommandedEEPose"`:

```yaml
action_topics:
  "/ee_pose_right":
    arm: "right"
    msg_type: "CommandedEEPose"   # else defaults to Float64MultiArray joint parsing
```

Task description: write it to not imply a fixed order/sequence unless the
demonstrations actually follow one consistently — check a few episodes' order
before assuming "A then B". Draft the wording and show it to the user for
approval before it goes in the config — don't finalize task description
phrasing unilaterally.

## Step 5 — Convert

```bash
uv run mcap-convert \
  --input-dir /srv/shared/datasets/Anvil_raw_data/<name> \
  --config configs/mcap_converter/<config>.yaml \
  --output-path /srv/shared/datasets/anvil/lerobot/<task>_<date>_<variant> \
  --fps 30 \
  --include-flagged critical \
  --skip-episode-idx "<corrupt positions>" \
  --task "<task description>"
```

**300+ episodes**: sequential conversion is CPU/decode-bound (~10-25s/episode
regardless of GPU — video *encoding* isn't the bottleneck, so `--vcodec h264_nvenc`
doesn't help here). Split into N parallel shards by episode-position range and merge:

```bash
# shard i keeps episodes [start,end) — skip everything else
uv run mcap-convert ... --output-path /tmp/shard_i --skip-episode-idx "1:start,end:"
# ...run all shards concurrently (nohup + disown), wait, then:
uv run merge-datasets /tmp/shard_1 /tmp/shard_2 ... --output <final_path>
```

This gave ~6-8x wall-time reduction in practice (24-core machine, 6-8 shards).

**Deriving a variant from an already-converted dataset** (e.g. dropping a camera,
slicing state/action dims) is much faster than reconverting from raw MCAP — but
only if done safely:

```bash
cp -al <source_dataset> <new_dataset>   # hardlinks videos, instant, zero extra disk
```

Then edit only the parquet/JSON files that actually change, using
**write-to-temp-then-`os.replace()`**, never `open(path, "w")` directly:

```python
tmp = path + ".tmp"
df.to_parquet(tmp, index=False)
os.replace(tmp, path)   # atomic rename breaks the hardlink safely
```

`open(path, "w")` on a hardlinked file truncates the *shared inode* — it silently
corrupts the source dataset too, not just the copy. This has happened once in this
repo; the fix cost a full reconversion.

## Step 6 — Validate and card

```bash
uv run dataset-valid --root /srv/shared/datasets/anvil/lerobot/<name>
```

**Checkpoint — before running conversion:** show the finished config (or a
diff from the template) to the user and get a go-ahead, especially for
`--skip-episode-idx` and `--include-flagged` choices — these silently drop
episodes if wrong.

Then write `dataset_card.yaml` next to it — fields: `name`, `task`, `description`,
episodes/frames/fps, cameras, native resolution, `state_dim`/`action_dim`,
`task_description`, `config` path, `split`, `status`, `owner`, `created_at`,
and a `notes:` block documenting anything non-obvious (corrupt episodes excluded,
derivation history, known data-quality caveats). Future readers trust this file
more than the raw directory listing — keep it accurate when a dataset gets
regenerated.

## Common Mistakes

| Symptom | Cause | Fix |
|---|---|---|
| "All episodes produced 0 frames" | Quality report flags every episode critical for a topic the config doesn't use | `--include-flagged critical`, after confirming the flag really is a false positive |
| Conversion crashes with `RecordLengthLimitExceeded` on the same episode every retry | Genuinely corrupted MCAP (truncated write), not a race condition | `--skip-episode-idx` at that episode's sorted *position*, not folder name |
| A derived/sliced dataset silently corrupts the dataset it was copied from | `cp -al` + `open(path,"w")` on a hardlinked file | write-temp + `os.replace()` instead |
| pi0.5 (or any QUANTILES-normalized policy) diverges early in training | A held-position/idle arm's dims have near-zero variance; QUANTILES divides by `q99-q01` | Don't include the unused arm in `arms:` at all (8-D, not 16-D with padding) |
| `mcap-convert` skips a big chunk of episodes with no error | Default `--include-flagged warning` skips `critical`; check the report first | Confirm severities before choosing the flag threshold |
