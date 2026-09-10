# Checkpoint intake

What to do when a new weight lands, before it ever drives the arm.

**Why this exists:** almost every way a checkpoint and its config can disagree
fails *silently*. Nothing raises. A missing camera becomes a grey frame, an
extra one hangs startup forever, a wrong action slice publishes zeros, corrupt
weights load fine and emit garbage. So we verify against the checkpoint's own
metadata rather than trusting a filename, a note, or the last config that
worked.

---

## Quick path

```bash
# 1. What shape is it, and which config fits?
./.venv/bin/python scripts/suggest_shape_config.py <checkpoint_dir>

# 2. Confirm the pick (weights scan + full contract check)
./.venv/bin/python scripts/preflight_checkpoint.py <checkpoint_dir> \
  --config configs/lerobot_control/shapes/<picked>.yaml

# 3. Actually run the model — real forward pass, no robot (see step 5)
docker run --rm --gpus all -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf:ro -v "$PWD":/workspace/repo:ro \
  -v "$PWD/model_zoo":/model_zoo:ro -w /workspace/repo \
  ghcr.io/anvil-robotics/lerobot-inference:latest \
  python3 scripts/offline_inference_test.py /model_zoo/<ckpt> --config <picked>

# 4. Point .env at both, then launch
```

Step 2 must exit 0. If it does not, fix what it reports — do not launch anyway.

---

## Shape configs

`configs/lerobot_control/shapes/` holds one config per **robot shape**, named
`<arms>arm_<cameras>cam[_variant]`. They are task-agnostic and reusable: pick
the one matching your checkpoint, no editing.

| Config | State | Cameras | Publishes |
|---|---|---|---|
| `1arm_2cam` | 8 | chest, wrist_r | right `[0:8]` |
| `1arm_3cam` | 8 | chest, wrist_l, wrist_r | right `[0:8]` |
| `1arm_3cam_waist` | 8 | chest, waist, wrist_r | right `[0:8]` |
| `2arm_3cam` | 16 | chest, wrist_l, wrist_r | left `[0:8]`, right `[8:16]` |
| `2arm_3cam_rightonly` | 16 | chest, wrist_l, wrist_r | right `[8:16]` + left pinning |
| `2arm_4cam` | 16 | chest, waist, wrist_l, wrist_r | left `[0:8]`, right `[8:16]` |
| `2arm_4cam_rightonly` | 16 | chest, waist, wrist_l, wrist_r | right `[8:16]` + left pinning |

They deliberately **omit `model.task_description`**, so it comes from the
checkpoint's own `anvil_config.json` (`inference_node.py:231-234`). That is what
makes one config serve many tasks.

> **Not every checkpoint has one.** As of 2026-09-10, 10 of 19 checkpoints in
> `model_zoo/` were trained without a `task_description`, including
> `smolvla-flip/005000` and everything above it, and all of
> `flip_pack_20260827_single_arm`. A VLA policy cannot run without one — the
> node refuses to start (`inference_node.py:382`).
>
> For those, uncomment the line in the shape config and set it verbatim:
>
> ```yaml
> model:
>   task_description: "Flip the package upside down."
> ```
>
> `suggest_shape_config.py` reports `task_description : MISSING` when this
> applies, and `offline_inference_test.py` fails with the same message rather
> than a traceback. Get the string from a sibling checkpoint of the same run
> (`smolvla-flip/001500` and `002500` have it) or from the training command.

No shape matches? Copy the closest, edit `arm_mapping` and `cameras.mapping`,
name it for its shape, and add a row above.

The older task-named configs (`inference_flip.yaml`, `inference_default.yaml`,
…) still work and are unchanged. Prefer a shape config for anything new.

### Choosing between `_rightonly` and the plain variant

Both have the same state width and cameras, so `config.json` cannot tell them
apart. The difference is a property of the **training data**: does the left arm
actually move?

`suggest_shape_config.py` answers this from the normalizer stats. A static joint
has a degenerate spread that the dataset stats floor clamps to a fixed minimum,
and the signature is the *same* minimum repeating across dims — independent
joints never land on an identical spread by chance.

- Static left arm → **`_rightonly`**. `action[0:8]` was trained to reproduce the
  current observation ("hold position"), not as a target. Tracking it makes the
  controller chase a statistical mean pose that is not where the arm is;
  `max_relative_target` caps each step but the error accumulates, so the arm
  creeps.
- Both arms move → the **plain** variant.

---

## The seven checks

### 1. Read the checkpoint's own declarations

`config.json` — policy type, camera keys, `observation.state` / `action` widths,
`chunk_size`, `normalization_mapping`. `anvil_config.json` — `action_type` and
the exact `task_description`. `train_config.json` — dataset and step count.

Never trust the directory name or an inherited comment. A checkpoint here was
labelled "single-arm 8 DOF" in `.env` while `config.json` said 16-DOF bimanual.
The file wins.

### 2. Scan the weights for NaN/Inf

`preflight_checkpoint.py` does this. It has caught real corruption twice, both
times from a bad **copy** rather than a bad export — and the two bad copies had
NaN in *different* tensors, which is how we knew it was the transfer.

A corrupt checkpoint loads without error. If this fails, re-copy and re-scan;
check `dmesg` for I/O errors if it recurs.

### 3. Run preflight against the config

Cross-checks camera keys, state width vs `arm_mapping × model_joint_order`, the
`action_start:action_end` slice, and `task_description`. Exit 0 or do not launch.

### 4. Read the normalizer stats per joint

Not covered by preflight, and worth doing for any bimanual or new-rig
checkpoint:

```bash
./.venv/bin/python -c "
from safetensors.numpy import load_file; import numpy as np
d=load_file('<ckpt>/policy_preprocessor_step_*_normalizer_processor.safetensors')
print(d['observation.state.std'])"
```

Look for **degenerate dims** — a `std` or `q99-q01` sitting at the stats floor.
Those are static joints turned into high-gain noise channels. See
[Pinning](#state-pinning).

### 5. Check runtime dependencies

- Tokenizer cached under `HF_CACHE` if `HF_HUB_OFFLINE=1`
  (pi0/pi0.5 → PaliGemma; SmolVLA → SmolVLM2).
- `LEROBOT_EXTRAS` includes the policy family (`pi`, `smolvla`). Changing it
  needs `docker compose build`.

### 6. Wire up `.env`

`MODEL_PATH` and `CONFIG_FILE` must be the pair preflight approved.

### 7. Offline inference test

`preflight_checkpoint.py` compares metadata. `offline_inference_test.py`
actually loads the weights and runs the policy on synthetic inputs, which
catches what static checks cannot:

- processor pipelines that fail to construct
- a tokenizer missing from `HF_CACHE` under `HF_HUB_OFFLINE=1`
- VRAM and **host RAM** that do not fit
- real per-chunk latency
- dead outputs (identical action for every input) or values far outside the
  training range

Run it in the inference image, not the host venv — the host `.venv` has no
`transformers`, so the VLA policies cannot even import there.

Measured on the RTX 5090 Laptop (24 GiB), 2026-09-10:

| Checkpoint | Load | Steady | Chunk | VRAM |
|---|---|---|---|---|
| `smolvla_flip_pack_20260904_plus_005000` | 6.8 s | 128 ms | 50x8 | 0.91 GiB |
| `smolvla-flip/005000` | 7.2 s | 113 ms | 50x16 | 0.91 GiB |
| `pi05-flip/005000` | — | — | — | **OOM, see below** |

**pi0.5 does not load on this machine.** `from_pretrained` is killed (SIGKILL,
exit 137) partway through loading the 9.3 GB checkpoint, with ~25 GiB host RAM
free. It is host RAM, not VRAM — the weights are materialised on CPU before the
move to GPU. If you need pi0.5 here, free memory first (swap was already 4 GiB
deep in the observed run) or load it on a machine with more RAM.

### 8. First run with monitoring

```bash
MONITOR_ENABLE=true
```

Writes per-step CSV. Preflight is a *static* contract check — it confirms the
checkpoint and config agree, not that cameras actually publish at 30 fps or that
the policy performs. Watch the gripper dim: pinned at a constant is the classic
symptom of a config problem, not a model problem.

---

## State pinning

`state_pinning` freezes a static arm's state dims at the training median instead
of feeding live encoder values.

**The problem.** A joint that never moves has a degenerate spread, floored by
the dataset stats to a fixed minimum. Normalisation then runs at enormous gain
on those dims — ~67 units/rad under `QUANTILES` with floor 0.03, against ~2 for
a joint that moves, roughly 35×. Under a degree of difference between the
training rest pose and the deployed one leaves normalised `[-1, 1]` entirely,
and `normalize_processor.py:377` does not clamp.

**Why it hurts pi0.5 specifically.** pi0.5 bins normalised state into 256 levels
and splices them into the **text prompt** (`processor_pi05.py:74-82`). An
out-of-range dim saturates to bin 0/255 or emits token `-1` — a string never
produced in training — corrupting the prefix that conditions the *entire* action
chunk, both arms. Observed: right gripper pinned at 0.048–0.053 instead of
reaching -0.003.

**Why SmolVLA survives the same data.** Its normalizer runs *after* the
tokenizer, and state reaches the model through `nn.Linear`
(`modeling_smolvla.py:693`). An out-of-distribution state is a large float that
degrades smoothly, not a bad token. This is why `smolvla-flip` behaved while
`pi05-flip` did not, on the same dataset.

**The fix.** Pinning to `q50` (or `mean`) puts those dims at exactly normalised
0.0, so the state portion of the prompt is byte-identical every step.

```yaml
state_pinning:
  enabled: true
  arms: [l]
  source: q50    # q50 for QUANTILES, mean for MEAN_STD
```

Set `enabled: false` to feed live state again for an A/B. Pinning also removes a
silent failure: `multi_process.py` falls back to `0.0` for any joint missing
from `/joint_states` (left arm unpowered, say), which with a floored std
normalises to -100 or worse with no warning.

The long-term fix is retraining with the static arm dropped from state and
action. Pinning is an inference-side mitigation.

---

## Known silent failures

| Symptom | Cause |
|---|---|
| Inference never starts, no useful log | Config lists a camera the checkpoint does not declare — `has_complete_observation()` blocks forever |
| Arm moves plausibly but wrong; decays to a constant pose | Camera missing or misspelled — substituted with a blank -1 image, mask 0 (`modeling_pi05.py:1195-1204`) |
| Arm reaches correctly, gripper never closes | The old `min_position_delta` deadband. Removed — it had no upstream equivalent and a plausible value froze the gripper |
| Gripper closes before the arm reaches the object | A scalar `max_relative_target` throttles the arm (radians) but exceeds the gripper's whole ~0.05 m travel, so it shuts in one step. Use a per-joint mapping with `finger_joint1: 0.005` |
| Values land in the wrong joints | State width mismatch — pi0/pi05 pad to `max_state_dim` 32 and truncate, absorbing the error |
| Overhanging joints publish zeros | `action_end` past the real action width |
| Left arm creeps to a pose it was never in | Bimanual checkpoint with a static left arm, published as a target. Use `_rightonly` |
| Garbage actions from a checkpoint that loads fine | NaN/Inf in weights — always from a bad copy so far |
| Subtly degraded actions, nothing in the log | `task_description` reworded. VLA policies condition on it |
| Node refuses to start, "has no task_description" | Checkpoint trained without one *and* the config does not set it. Not silent, but common — 10 of 19 checkpoints here |
| Container dies during load, exit 137, no traceback | Host-RAM OOM. pi0.5 needs more than this machine has free |

---

## Adding a shape config

`scripts/` has no generator committed; the shape files are plain YAML and
self-contained (the config is bind-mounted as a single file, so there is no
include mechanism). To add one:

1. Copy the closest existing shape file.
2. Edit `arm_mapping` (state width = `len(arm_mapping) × len(model_joint_order)`)
   and `cameras.mapping`.
3. Set each arm's `action_start`/`action_end` to its slot in
   `sorted(arm_mapping)` order — this must match the assembly in
   `multi_process.py:216-227`.
4. Name it `<arms>arm_<cameras>cam[_variant].yaml`, add a table row above.
5. Verify against a real checkpoint with `preflight_checkpoint.py`.
