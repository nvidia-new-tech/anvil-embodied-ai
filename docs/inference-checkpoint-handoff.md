[← Back to README](../README.md) · [Run Inference](inference.md)

# Checkpoint → Inference Handoff

[`inference.md`](inference.md) covers how to *run* the stack: `.env`, script flags, DDS, topologies.
This document covers the step before that — taking a checkpoint out of `model_zoo` and
working out the inference config it actually requires.

**Why this needs its own document:** every way a checkpoint and its inference config can
disagree fails *silently* on the robot. Nothing raises, nothing warns, and the arm still
moves plausibly. The failures we spent 2026-08-20 → 09-04 chasing were all of this class,
not one of them a model-quality problem.

---

## 1. Run the preflight check first

Before the arm is powered:

```bash
uv run scripts/preflight_checkpoint.py <checkpoint_dir> --config <inference_yaml>
```

It reads the checkpoint's own `config.json` + `anvil_config.json` and cross-checks them
against the YAML. Exit code 1 means do not launch.

```bash
# example
uv run scripts/preflight_checkpoint.py \
    /srv/shared/model_zoo/anvil/macp_20260813__pass/pi05_bottle_handoff/checkpoints/last \
    --config configs/lerobot_control/inference_flip.yaml

# weights only, before you have a config to compare against
uv run scripts/preflight_checkpoint.py <checkpoint_dir> --skip-weight-scan
```

What it verifies, and the failure each check prevents:

| Check | If wrong, what actually happens |
|---|---|
| Camera set matches `observation.images.*` | A camera the checkpoint declares but the config omits is substituted with a **blank −1 image, attention mask 0** (`modeling_pi05.py:1195-1204`). No error. The policy looks at a grey frame and decays toward a constant prior pose. |
| No extra cameras in the config | A camera only the YAML lists makes `ObservationManager.has_complete_observation()` **wait forever** for a feature with no slot. Inference never starts and the log says nothing useful. |
| `observation.state` width | pi0/pi05 **pad state to `max_state_dim` (32) and truncate the action back down**, so an 8-DOF checkpoint fed 16 DOF reads left-arm values in right-arm slots and still produces plausible-looking motion. |
| `arms.*.action_start/end` within the real action width | The overhang silently **publishes zeros**. |
| Slice width == `model_joint_order` length | Joints land off-by-N inside the slice. |
| `task_description` matches `anvil_config.json` | The language tokens are a **conditioning input**. A reworded string degrades actions and reports nothing. |
| Weights contain no NaN/Inf | A NaN checkpoint produces motion, just wrong. |
| `queue_trigger_threshold` ≤ `chunk_size` | The queue can never hold that many, so inference re-triggers every step. |
| `safety.min_position_delta` scale | See §5 — a plausible-looking value freezes the gripper. |

---

## 2. What the inference side needs from a `model_card.yaml`

The PRO6000 shared-training-server plan proposes a `model_card.yaml` next to every training
output (see `docs/pro6000.md` once it lands). For a card to be *sufficient for deployment*,
it has to carry the fields inference cannot guess. The
training-oriented fields (`steps`, `batch_size`, `created_by`) do not determine any of them.

Minimum additions we would ask for:

```yaml
# --- inference contract (in addition to the training fields) ---
cameras: [chest, wrist_l, wrist_r]   # EXACT observation.images.* keys, verbatim
state_dim: 16
action_dim: 16
arms:                                 # which action slice drives which arm
  left:  {action_start: 0,  action_end: 8,  driven: false}
  right: {action_start: 8,  action_end: 16, driven: true}
action_type: absolute                 # absolute | delta_obs_t | delta_sequential
task_description: "Flip the package upside down."   # verbatim, conditioning input
chunk_size: 50
n_action_steps: 50
stats_floor: 0.03                     # if the dataset stats were floored — see §4
inference_config: configs/lerobot_control/inference_flip.yaml   # known-good config
```

Three of these are load-bearing and worth stating explicitly on the card:

- **`cameras`** — the single highest-risk field. Must be the literal
  `observation.images.<name>` suffixes. Sending an *extra* camera is ignored by the model;
  omitting or misspelling one is the fatal direction.
- **`arms[].driven`** — for a bimanual checkpoint trained on a task where one arm never
  moves, that arm's action slice is "hold position", **not** a target to track. Tracking it
  drives the idle arm toward a statistical mean pose. See `model_zoo/Pi05 Inference Note.md` §3.
- **`stats_floor`** — records that the checkpoint depends on modified normalization stats.
  See §4.

> Recording `inference_config` — the config the checkpoint was last known to run correctly
> under — is the cheapest single thing that makes a checkpoint re-deployable by someone who
> was not there when it was trained.

---

## 3. Deriving the config from the checkpoint

Field-by-field, where each inference YAML value has to come from:

| Inference YAML | Source of truth | Notes |
|---|---|---|
| `cameras.mapping` values | `config.json` → `input_features` `observation.images.*` | Names verbatim. |
| `joint_names.arm_mapping` | `observation.state` shape ÷ joints per arm | Single-arm checkpoint ⇒ **one** key only. |
| `joint_names.model_joint_order` | dataset converter config | Must match training order. |
| `arms.*.action_start/end` | `output_features.action` shape + arm layout | Single-arm: `action[0:8]`, **not** `[8:16]`. |
| `model.task_description` | `anvil_config.json` | Leave `null` to auto-read — recommended. |
| `inference_tuning.rtc.*` | tuned on hardware | `queue_trigger_threshold` ≤ `chunk_size`. |

**The single-arm vs bimanual trap.** A bimanual checkpoint feeds 16-DOF state and publishes
`action[8:16]`. A single-arm checkpoint declares `observation.state [8]` / `action [8]`, so
`arm_mapping` lists one arm and the slice is `action[0:8]`. Copying the bimanual config and
only swapping the model path is silently wrong in both directions at once. See
`configs/lerobot_control/inference_singlearm_flip.yaml` for a worked comparison.

---

## 4. Normalization: degenerate state dims

The failure mode that cost the most time, and the one most likely to recur on the next
task where one arm is idle.

**Setup.** A joint that never moves in the training data has a degenerate `q99 − q01`. The
dataset stats floor widens it to `0.03`. QUANTILES normalisation then runs at
`2 / 0.03 ≈ 67` units/rad on those dims, against 1.3–2.6 on a joint that actually moves —
roughly 35× more sensitive. A sub-degree difference between the training rest pose and the
deployed one pushes the dim outside normalised `[-1, 1]`, and `normalize_processor.py:377`
**does not clamp**.

**Why SmolVLA survives it and pi0.5 does not.** SmolVLA's normalizer runs *after* the
tokenizer and state reaches the model through `nn.Linear` (`modeling_smolvla.py:693`) — the
result is a large float and it degrades smoothly. pi0.5 bins the normalised state into 256
levels and splices them into the **text prompt** (`processor_pi05.py:74-82`). Out of range
saturates to bin 0/255 or emits token `-1`, a string never produced during training,
corrupting the prefix that conditions the entire chunk — **both arms, not just the pinned
dims.** This is exactly why `smolvla-flip` behaved and `pi05-flip` did not, on the same dataset.

**Observed symptom:** right gripper pinned at 0.048–0.053 instead of reaching −0.003.
Deployment pose left j2 = −0.194 rad normalises to −1.84 and tokenises to `-1`.

**Mitigation — `state_pinning`.** Freeze the idle arm's state dims at the training median
instead of feeding live encoder values:

```yaml
state_pinning:
  enabled: true
  arms: [l]        # arm_mapping keys to pin
  source: q50      # q50 | mean, read from the checkpoint's own normalizer stats
  # values: [...]  # optional explicit override, one per pinned dim
```

`q50` lands at exactly normalised `0.0` (token 128) on every pinned dim, so the state
portion of the prompt becomes byte-identical every step. Values are read from the
checkpoint's `policy_preprocessor.json` → `normalizer_processor` safetensors, so they cannot
drift from the weights.

This also closes a silent path: `multi_process.py` previously defaulted any joint missing
from `/joint_states` to `0.0`. For left j4 (`q01 = 1.574`) that normalises to **−105.9** with
no warning.

Set `enabled: false` to feed live state again and A/B against SmolVLA.

> **Never recompute normalization on the inference side.** The stats are baked into the
> checkpoint; going through `make_pre_post_processors(pretrained_path=<ckpt>)` is correct.
> Recomputing from the dataset picks up *unfloored* stats and the idle arm's output blows up.

---

## 5. Other silent failures worth knowing

**`safety.min_position_delta` mixes units.** `ActionLimiter` applies one scalar threshold to
a vector that mixes radians (j1–j7) with the **PRISMATIC** gripper in metres (0–0.05 m
travel). Anything at the 0.01+ scale freezes the gripper: the arm reaches, then never
closes. Use `null`, or ~0.002 if friction genuinely demands a deadband.

**Image resolution.** Frames are stored 640×480; the model resizes to 224×224 internally.
Send native resolution — do not pre-scale.

**Idle-arm action dims.** `preflight_checkpoint.py` warns when action dims are never
published. That is *deliberate* for a bimanual checkpoint whose idle arm must not be
tracked, and a bug in every other case. Read the warning, don't silence it.

---

## 6. Suggested flow for a checkpoint arriving from PRO6000

```bash
# 1. Checkpoint lands in shared storage on PRO6000
/srv/shared/model_zoo/anvil/<dataset>/<run>/checkpoints/last

# 2. On the workcell: preflight against the config you intend to use.
#    MODEL_PATH must be absolute or start with ./ — bare relative paths
#    are treated as Docker named volumes.
uv run scripts/preflight_checkpoint.py <ckpt> --config <yaml>

# 3. Fix every error. Read every warning.

# 4. Fake hardware — no robot, validates DDS + the full pipeline
MODEL_PATH=<ckpt> ./scripts/run_inference.sh --fake-hardware --profile inference up --build

# 5. Real hardware, monitor on
MODEL_PATH=<ckpt> ./scripts/run_inference.sh --monitor-enable up --build
```

Steps 2–4 need no robot time. Every failure in §1 and §4 is catchable there.

---

[← Back to README](../README.md) · [Run Inference](inference.md)
