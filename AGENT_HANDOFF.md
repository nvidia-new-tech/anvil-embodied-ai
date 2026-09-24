# Agent Handoff — anvil-embodied-ai

Comprehensive context for an AI agent picking up work in this repo. Read this
before touching data, training, or inference code — several sections
document real incidents (data corruption, silent config bugs) whose fixes are
easy to accidentally revert if you don't know they happened.

## What this repo is

OpenArm robot embodied-AI stack: MCAP → LeRobot dataset conversion
(`mcap_converter`), VLA policy training (`anvil_trainer` wrapping
`lerobot-train`: SmolVLA, pi0.5/pi05, GR00T N1.7), and ROS2 real-robot
inference (`ros2/src/lerobot_control`).

Two persistent conventions to know immediately:
- **Shared storage** at `/srv/shared/` (PRO6000 training server) holds
  canonical datasets (`/srv/shared/datasets/anvil/lerobot/<task>_<date>_<variant>/`),
  raw sessions (`/srv/shared/datasets/Anvil_raw_data/`), and validated
  models (`/srv/shared/model_zoo/anvil/<dataset>/<policy>_<dataset>_downstream/`).
- **Git remotes**: `origin` = anvil-robotics upstream, `nvidia` = nvidia-new-tech
  fork (team push target). Push to `nvidia`, not `origin`, unless told otherwise.

## Skills — use these, don't reinvent the pipeline

`.claude/skills/mcap-to-dataset/SKILL.md` and
`.claude/skills/train-checkpoint/SKILL.md` are the canonical, tested
procedures for the two most common tasks in this repo (raw MCAP → dataset,
dataset → checkpoint → model_zoo). They encode every real bug this repo has
hit (see Incident History below) as an explicit step or a Common Mistakes
table entry. If you're about to write a mcap_converter config or launch
`anvil-trainer` by hand, read the matching skill first.

Both skills now have **mandatory ask-checkpoints**: at defined points (camera/
arm/action-space selection, task description, launch hyperparameters,
promotion) the skill requires literally stopping and asking the user via
AskUserQuestion, not proceeding on your own judgment and disclosing it after
the fact. This was deliberately tested: a first version of this instruction
was silently overridden by auto-mode/autonomous-execution framing in three
separate test runs (agents proceeded and only mentioned the skip in a final
summary). If you are being run autonomously and reach one of these
checkpoints, the checkpoint still applies — surface the question to whoever
dispatched you rather than resolving it yourself, even under time pressure or
a "just testing" framing.

## Architecture map

```
packages/mcap_converter/   MCAP -> LeRobot dataset conversion
  core/extractor.py          joint/EE-pose parsing, action extraction
  core/aligner.py             timestamp alignment across topics
  core/writer.py              LeRobot dataset feature/metadata writing
  config/schema.py            DataConfig, ActionTopicConfig (msg_type field)
  cli/convert.py               mcap-convert entrypoint
  cli/mcap_valid.py            quality scan (mcap-valid)

packages/anvil_trainer/     anvil-trainer CLI wrapping lerobot-train
  config.py                   policy config building; _VLA_POLICIES set

configs/mcap_converter/*.yaml   per-task conversion configs (arms, cameras,
                                  action topic/msg_type, resolution)
configs/lerobot_control/         inference configs
  shapes/*.yaml                  per-checkpoint shape configs (camera set,
                                  state/action dims) for preflight checking

ros2/src/lerobot_control/    real-robot ROS2 inference node
  inference_node.py             policy loading, publishers, per-step inference
  action_limiter.py             joint-space rate/clamp limiting
  ee_pose_limiter.py             EE-space rate/clamp limiting (newer)
  strategies/multi_process.py    inference process strategy

ros2/src/anvil_msgs/          CommandedEEPose.msg (EE-space action message)

scripts/
  preflight_checkpoint.py       checkpoint/inference-config compatibility check
  offline_inference_test.py     dry-run policy inference on recorded data
  analyze_grasp_timing.py, analyze_tracking_error.py, check_obs_distribution.py
```

## Data representations — the two action spaces

- **Joint-space** (default, most checkpoints): 8-D per arm (7 joints +
  gripper), `std_msgs/Float64MultiArray`, published via
  `/follower_{arm}_forward_position_controller/commands`.
- **EE-space** (Cartesian, newer): 8-D (`pos_xyz` + `quat_xyzw` + gripper),
  `anvil_msgs/CommandedEEPose`, published to `/commanded_ee_{arm}`. Requires
  `msg_type: "CommandedEEPose"` in the mcap_converter action_topics config,
  and the real robot's `ARMS_CONTROL_CONFIG_FILE` to have `commanded_ee: true`.
  Built specifically to test whether Cartesian action fixes an approach-depth/
  empty-grasp failure mode seen on joint-space checkpoints (see Known Issues).
  **Confirmed working on real hardware** as of the `pack_pick_place_20260911_ee`
  5k checkpoint — this validates the EE-space inference path end-to-end
  (`ee_pose_limiter.py`, `CommandedEEPose` publishing in `inference_node.py`).

Single-arm vs bimanual config: **do not declare an arm in `arms:` if it has
zero real driven data** — `parse_joint_name` silently drops unmapped arm
prefixes, so declaring a genuinely-unused arm creates a phantom near-zero-
variance dimension that breaks `QUANTILES` normalization at training time.
Only declare both arms when one is idle-but-present (real robot, real
joint_states, just not commanded this task) — that case uses the
observation-as-action fallback (hold position) rather than omission.

## Incident history (read before repeating)

1. **Flip-pack 0-frames bug**: single-arm task on a bimanual rig had no
   action topic for the idle arm → every episode produced 0 frames. Fixed
   with observation-as-action fallback in `aligner.py`/`extractor.py`.
2. **GR00T backbone injection bug**: `anvil_trainer/config.py`'s
   `_VLA_POLICIES` set excluded `"groot"`, so ResNet backbone flags got
   injected into a policy config that doesn't have those fields
   (`draccus.DecodingError`). Fixed by adding `"groot"` to the set — don't
   remove it.
3. **Major dataset corruption**: a derived dataset was built with
   `cp -al` (hardlink) then `open(path, "w")` on the parquet/JSON files.
   Hardlinks share inodes, so this truncated the **source** dataset too
   (16-D → 8-D data loss, undetected until training broke). Fixed by full
   reconversion from raw MCAP. **Any hardlink-based dataset derivation must
   use write-to-temp + `os.replace()`, never `open(path, "w")` directly** —
   this is now called out in the mcap-to-dataset skill's Common Mistakes.
4. **writer.py action-naming bug**: `_define_features()` always reused
   `observation.state`'s joint names for the action feature metadata, even
   when the action was `CommandedEEPose` — action *values* were correct but
   metadata *names* were wrong (showed joint names instead of `pos_x` etc).
   Fixed with an explicit `uses_ee_pose_action` branch.
5. **pi0.5 underperforming SmolVLA on real robot**: root-caused to
   `train_expert_only=true` being used for memory savings — official LeRobot
   docs state this explicitly as "less memory, reduced success rate." Fixed
   recipe: `train_expert_only=false` + `gradient_checkpointing=true` at
   `batch_size=16` (validated on a 96GB card, 3 cameras). This is the
   validated recipe in the train-checkpoint skill — don't reach for
   `train_expert_only=true` as a default.
6. **GR00T N1.7 vs N1.5 incompatibility**: `anvil_trainer`'s bundled LeRobot
   0.5.1 only supports GR00T N1.5's config shape; N1.7 needs LeRobot 0.6.1,
   which breaks `anvil_trainer`'s monkey-patches. Resolved with a separate
   `.venv-groot` (0.6.1) alongside the main `.venv` (0.5.1) — don't try to
   unify these without re-testing anvil_trainer's patches against 0.6.1.
7. **pi0.5 divergence from degenerate `QUANTILE10` stats** (flip-pack,
   2026-08-28): pi0.5's native normalization divides by `q99 − q01`. A
   hold-position fallback arm (see single-arm convention above) has
   near-constant values, so that spread is ~0 for its dimensions — one
   dimension's `q99-q01` was `5.69e-06`, amplifying a small input by
   **175,867×**. Loss diverged from a healthy 0.04–0.07 to 6,076 by step 900,
   grad norm into the hundreds of thousands. `MEAN_STD` alone would not have
   saved it either (min std 1.02e-04, still ~9,800× amplification). Fixed
   with `scripts/floor_degenerate_stats.py --floor 0.03`, which widens
   degenerate dimensions symmetrically about `q50` (`stats.json.bak` backs up
   the original; `--restore` reverts). `0.03` was chosen deliberately — `0.1`
   also catches the real right-gripper signal (spread 0.053) and would halve
   it. **Any time a hold-position/idle-arm dimension coexists with
   `QUANTILES`/`QUANTILE10` normalization, check for this before training**,
   not just for SmolVLA's `MEAN_STD` case (item 3 in the mcap-to-dataset
   skill's Common Mistakes table only covers the "don't declare the arm at
   all" prevention — this is the same failure mode when the arm *must* stay
   declared, e.g. mid-training bimanual-compatible datasets).
8. **`delta_obs_t` (relative action) normalization mismatch** — unresolved,
   not just historical: `transforms.py:293-295` computes the delta
   representation inside the dataset transform, but `meta/stats.json` is
   still computed from **absolute** actions. Feeding delta-scale values (near
   0) through an absolute-scale normalizer (`q01 −0.96 … q99 2.10`) diverges
   (loss ~1.5e8 raw, ~700 even with the degenerate-stats floor from item 7
   applied — vs 0.103 for the equivalent absolute-action run). The floor hack
   cannot fix this because the problem isn't degenerate spread, it's the
   wrong reference frame entirely. **Real fix, not yet implemented:**
   recompute `meta/stats.json` over the delta-transformed actions, not the
   absolute ones. Don't attempt `delta_obs_t` training without doing this
   first.

## MCAP corruption recovery — what worked and what didn't

Seen on `flip-pack` (2026-08-27/28): 44 of 300 episodes (all in one contiguous
block, package C, `0248` onward) failed with `RecordLengthLimitExceeded` /
`EndOfFile`. Headers were intact and byte-identical to good files — only the
footer/closing magic was missing, consistent with the recorder or disk dying
mid-write partway through the session (corruption starts at one point and
never recovers for the rest of the session — check for this contiguous
pattern, it's diagnostic).

`mcap recover` (foxglove mcap-cli) salvaged 41/44, but the recovered
durations formed a monotonic staircase (11s → 2.1s) against a healthy
7.5–21.7s range — i.e. recovery returns a truncated prefix of the real
episode, not the full thing. **Decision made: discard all salvage, keep only
the clean episodes, re-record the lost block.** Reasoning: files under ~5s
cannot contain a complete task demonstration (truncated mid-motion), and even
the longer recovered files were cut at the *end* — meaning the task's closing
phase, often the hardest/most informative part, is exactly what's missing.
Training on truncated demonstrations teaches the policy to stop partway
through the task. **Don't reach for `mcap recover` output as usable training
data without checking recovered duration against the healthy distribution
first** — a file that "recovers" without erroring can still be silently
incomplete.

## Real-robot findings trace back to data distribution, not hyperparameters

flip-pack SmolVLA (5k checkpoint, best real-robot performer at the time)
showed three systematic weaknesses, all traced to gaps in the recorded
demonstrations rather than training issues:

| Symptom | Root cause in the demos |
|---|---|
| Flip not clean — upright packages catch on their long edge | Almost every demo had the package lying flat; the policy never saw the upright case |
| Large packages often don't complete the flip | Demos never lifted high for either package size — learned motion has too little clearance |
| Hesitant to slide under a ground-flush package even with sufficient gap | Demonstrators rarely jammed the gripper in hard during collection; policy inherited that caution |

**General lesson: when a real-robot failure mode looks systematic (not
random/noisy), check what the demonstrations actually contained before
tuning hyperparameters or blaming the model** — the fix for a distribution
gap is more/better data collection, not a different training recipe. This is
the same category of finding as the approach-depth/empty-grasp issue in Known
Issues below; that one is not yet root-caused to this level, so check demo
diversity for it too before assuming a training/architecture cause.

Related decision carried forward: **train straight to a target step count
(e.g. 5k) with no early stopping**, rather than stopping at best-val. pi0.5 on
this same task early-stopped at 6k with best-val at 3k, and that turned out
under-trained for real-robot competence — val loss was a misleading stopping
signal. This generalizes the anvil_trainer "no early stopping" caveat already
noted above: it's not just "anvil_trainer doesn't support it," it's that
early stopping via val loss has concretely produced a worse real-robot model
here more than once.

## Known unresolved / open threads

- **Approach-depth / empty-grasp issue** on the joint-space
  `pack_pick_place_20260911` SmolVLA checkpoint (5k, `candidate` status, see
  its `model_card.yaml`): real-robot testing found insufficient Z-axis
  approach depth and occasional empty grasps. Offline correlation/bias
  analysis found no obvious per-dimension training bias. The EE-space
  checkpoint built to test a Cartesian-action fix has since tested
  successfully on hardware, but this has not been independently re-verified
  against the *original* failure mode in a controlled A/B — treat "EE-space
  fixed it" as a strong signal, not a closed investigation.
- `.venv-groot` / GR00T N1.7 path exists but is a secondary, less-trodden
  path — expect friction if picking this back up (gated HF repo access for
  `nvidia/Cosmos-Reason2-2B`, LeRobot version pinning).

## Validated training recipes (see train-checkpoint skill for exact CLI)

- **SmolVLA**: `--policy.pretrained_path=lerobot/smolvla_base`,
  `--policy.load_vlm_weights=true`, `MEAN_STD` normalization,
  `--split-ratio=1,0,0` (full data, no split — not the trainer's own
  default), `batch_size=64`.
- **pi0.5**: `--policy.pretrained_path=model_zoo/pi05_base_local`,
  `dtype=bfloat16`, `train_expert_only=false`, `freeze_vision_encoder=false`,
  `gradient_checkpointing=true`, `optimizer_lr=5e-5`, `MEAN_STD`
  normalization (not pi0.5's own `QUANTILES` default), `batch_size=16`.

`anvil-trainer` has **no early stopping or best-checkpoint tracking** — pick
checkpoints by real-robot result, not val loss (val loss has picked the wrong
checkpoint before: a lower-loss earlier step performed worse on hardware than
a later, higher-loss one). Default to `--split-ratio=9,1` (val only, no test)
once you're past initial sanity-checking of a recipe — freeing that data for
training matters more than a test-split number nobody acts on, given val loss
is already known to be an unreliable stopping signal here.

**fps choice**: cameras in this rig record at ~60Hz; `--fps 30` (exact ÷2) is
the validated choice across every session converted so far, because action
command topics run well above 30Hz (50-77Hz typically), so nothing gets
undersampled, and it halves dataset size/training compute for no measured
quality loss.

**pi0.5 memory tuning**: `--batch_size=64` OOMs even on a 95GB card — LeRobot
0.5.1's pi0.5 implementation uses `eager_attention_forward` (transformers
5.x), which materializes the full `[B, heads, L, L]` attention matrix in
fp32; with 4 cameras the prefix reaches ~1274 tokens (4×256 image + 200 text
+ 50 action), so batch 64 does not fit. `batch_size=16` +
`gradient_checkpointing=true` is the validated combo (see recipe above).
Don't extrapolate peak memory linearly from an OOM message — the OOM point
reported is where it ran out, not the actual peak requirement.

**pi0.5 base weights**: `lerobot/pi05_base` on the Hub is a 13.8GB fp32
`model.safetensors`; if link speed is a bottleneck, a bf16 local copy is
tensor-identical (verified: 812/812 tensor names match, 0 shape mismatches,
sampled value diffs are exactly fp32→bf16 rounding, 5e-05–4e-04 abs). Its
`config.json` is OpenPI-style, not a LeRobot `PI05Config`, so
`model_zoo/pi05_base_local/` is built from the three small Hub config files
plus a symlink to the local weights file — don't expect the directory to be a
self-contained download, it's a hybrid.

## Process/infra gotchas

- Sandbox/session teardown can kill background training jobs for reasons
  unrelated to the job (host suspend, OOM elsewhere) — this isn't a crash.
  Resume from the last `save_freq` checkpoint; re-pass every non-default CLI
  flag on resume (flags are NOT inherited from the checkpoint's saved config).
- Check process liveness by exact PID (`ps -p <pid>`), not `pgrep -f` pattern
  matching or log-text grepping — both have produced false positives that
  triggered duplicate concurrent training runs against the same output dir,
  and false "training complete" detections on log files that get appended to
  across resumes (a stale prior-segment "End of training" line matches
  immediately even though the current segment is still running).
- `/tmp` gets wiped between sessions on this environment — put logs in a
  persistent `logs/` dir, not `/tmp`.
- Model promotion to `/srv/shared/model_zoo` requires real-robot validation,
  not just a clean training run — see any existing `model_card.yaml` for the
  expected fields (`status: candidate` until confirmed reliable).

## Where to look for more detail

- `docs/data-conversion.md` — mcap_converter config reference table.
- `dataset_card.yaml` / `model_card.yaml` next to any dataset/checkpoint —
  always check these before trusting a directory listing; they document
  known caveats and derivation history.

This file is the single consolidated record of session-learned experience for
this repo — past per-date `SESSION_LOG_*.md` / work-log files have been
merged in here and removed to avoid the same lesson existing in two places
and drifting out of sync. **When you learn something new that would have
caused a repeat mistake if undocumented, add it here** (as a new Incident
History entry, or under the relevant existing section) rather than starting a
new dated log file.
