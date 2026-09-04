[← Back to README](../README.md)

# Run Inference

All inference scenarios go through `scripts/run_inference.sh`.

> **Deploying a checkpoint you did not train?** Read
> [Checkpoint → Inference Handoff](inference-checkpoint-handoff.md) first. This document
> covers running the stack; that one covers working out the config a given checkpoint
> requires, and the silent failure modes when the two disagree. Start with
> `scripts/preflight_checkpoint.py`.

**Start by copying `.env.example` to `.env` and editing it for your setup:**

```bash
cp .env.example .env
# then edit .env
```

### `.env` Variables

`.env` is the primary configuration file — set it once and reuse across runs. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | Yes (inference) | Host path to checkpoint dir. Must be absolute or start with `./` — bare relative paths are treated as Docker named volumes. |
| `ROS_DOMAIN_ID` | Yes | ROS2 domain ID — must match the Anvil Devbox. Leave empty for localhost-only. |
| `CYCLONEDDS_URI` | Yes | Path to CycloneDDS XML config (e.g. `configs/cyclonedds/two_pc_gpu.xml`). |
| `LEROBOT_EXTRAS` | VLA only | Comma-separated policy extras built into the Docker image — e.g. `smolvla`, `pi,smolvla`. **Rebuild the image after changing:** `docker compose build`. ACT and Diffusion leave this empty. |
| `HF_CACHE` | VLA only | Host path to HuggingFace model cache (default: `~/.cache/huggingface`). Required for Pi0, Pi0.5, SmolVLA — they load the PaliGemma tokenizer at runtime. |
| `CONFIG_FILE` | Yes | Path to inference config YAML (default: `configs/lerobot_control/inference_default.yaml`). |
| `ACTION_TYPE` | No | Action type passed to the **inference monitor node** (`inference_monitor_node`) only. The main inference node always reads this from `anvil_config.json` in the checkpoint via `resolve_action_type()` — this env var does **not** override it. Options: `absolute` · `delta_obs_t` · `delta_sequential`. |
| `ECHO_TOPIC_ONLY` | No | `true` = skip model loading, subscribe topics and log FPS only. For verifying DDS connectivity without a GPU or checkpoint. Equivalent to `--echo-topic-only`. |
| `MONITOR_ENABLE` | No | `true` = enable the inference monitor node (records per-step CSV + PNG report). Equivalent to `--monitor-enable`, but without the auto-plot on exit and output dir pre-creation that the flag provides. |
| `DEBUG` | No | `true` = enable extra metrics: action smoothness, queue depth stats, Action FPS. |

For full descriptions and defaults, see [`.env.example`](../.env.example).

### Script Flags

The script flags are a lightweight way to override behaviour at the command line without editing `.env`:

```bash
./scripts/run_inference.sh [--fake-hardware] [--monitor-enable] [--echo-topic-only] [--debug] [COMPOSE_ARGS...]
```

| Flag | What it does |
|------|-------------|
| `--fake-hardware` | Switches to `docker-compose.fake-hardware.yml` — simulates a 2-PC setup locally over a bridge network (CycloneDDS, no real robot). |
| `--monitor-enable` | Adds `--profile monitor` to the compose command. In production (non-fake-hardware) mode also exports `MONITOR_ENABLE=true`, pre-creates the output dir as the current user, and auto-plots the CSV on exit. |
| `--echo-topic-only` | Exports `ECHO_TOPIC_ONLY=true` — subscribes to topics and logs FPS without loading a model or GPU. Equivalent to setting `ECHO_TOPIC_ONLY=true` in `.env`. |
| `--debug` | Exports `DEBUG=true` — enables extra metrics: action smoothness, queue depth stats, Action FPS. Equivalent to setting `DEBUG=true` in `.env`. |

## Preflight: Match the Checkpoint to the Config

Do this before the arm is powered, especially for a checkpoint you did not train yourself.

**Every way a checkpoint and its inference config can disagree fails silently.** Nothing
raises, nothing warns, and the arm still moves plausibly. The 2026-08-20 → 09-04 debugging
was entirely this class of problem — not one case was model quality.

```bash
uv run scripts/preflight_checkpoint.py <checkpoint_dir> --config <inference_yaml>

# example
uv run scripts/preflight_checkpoint.py \
    /srv/shared/model_zoo/anvil/macp_20260813__pass/pi05_bottle_handoff/checkpoints/last \
    --config configs/lerobot_control/inference_flip.yaml

# weights only, before you have a config to compare against
uv run scripts/preflight_checkpoint.py <checkpoint_dir> --skip-weight-scan
```

It reads the checkpoint's own `config.json` + `anvil_config.json` and cross-checks them
against the YAML. **Exit code 1 = do not launch.**

| Check | What actually happens if it's wrong |
|-------|-------------------------------------|
| Camera set matches `observation.images.*` | A camera the checkpoint declares but the config omits is padded with a **blank −1 image and mask 0** (`modeling_pi05.py:1199-1202`). No error. The policy looks at a grey frame and decays toward a near-constant prior pose. |
| No *extra* cameras in the config | `has_complete_observation()` returns False forever, so inference never starts. `multi_process.py:306` builds a `"waiting for cameras: [...]"` string — but `get_incomplete_reason()` is **never called anywhere in the repo**, so that diagnostic is computed and discarded. The run just sits there. |
| `observation.state` width | pi0/pi05 pad state to `max_state_dim` (**32**, `configuration_pi05.py:41`) and truncate the action back down. An 8-DOF checkpoint fed 16 DOF reads left-arm values in right-arm slots and **still produces plausible motion**. |
| `arms.*.action_start/end` within the real action width | The overhang silently publishes **zeros**. |
| Slice width == `model_joint_order` length | Joints land off-by-N inside the slice. |
| `task_description` matches `anvil_config.json` | The language tokens are a **conditioning input** — pi0.5 splices them into the prompt. A reworded string degrades actions and reports nothing. |
| Weights contain no NaN/Inf | A NaN checkpoint still produces motion, just wrong. |
| `queue_trigger_threshold` ≤ `chunk_size` | The queue can never hold that many, so inference re-triggers every step. |
| `safety.min_position_delta` scale | See [Safety limits](#safety-limits) — a plausible value freezes the gripper. |

### Deriving the config from the checkpoint

| Inference YAML | Source of truth |
|----------------|-----------------|
| `cameras.mapping` values | `config.json` → `input_features` `observation.images.*`, **verbatim** |
| `joint_names.arm_mapping` | `observation.state` shape ÷ joints per arm — single-arm ⇒ **one** key |
| `joint_names.model_joint_order` | dataset converter config; must match training order |
| `arms.*.action_start/end` | `output_features.action` shape + arm layout |
| `model.task_description` | `anvil_config.json` — leave `null` to auto-read (recommended) |

> **Single-arm vs bimanual.** A bimanual checkpoint feeds 16-DOF state and publishes
> `action[8:16]`. A single-arm checkpoint declares `observation.state [8]` / `action [8]`, so
> `arm_mapping` lists one arm and the slice is `action[0:8]`. Copying the bimanual config and
> only swapping `MODEL_PATH` is silently wrong in **both** directions at once. See
> [`inference_singlearm_flip.yaml`](../configs/lerobot_control/inference_singlearm_flip.yaml)
> for a worked comparison.

> **Never recompute normalization on the inference side.** The stats are baked into the
> checkpoint, so `make_pre_post_processors(pretrained_path=<ckpt>)` is correct. Recomputing
> from the dataset picks up *unfloored* stats and an idle arm's output blows up — see
> [State pinning](#state-pinning-degenerate-state-dims).

### What a `model_card.yaml` must carry

Training-side fields (`steps`, `batch_size`, `created_by`) determine none of the above. For a
checkpoint to be deployable by someone who wasn't there when it was trained, the card needs:

```yaml
cameras: [chest, wrist_l, wrist_r]   # EXACT observation.images.* keys, verbatim
state_dim: 16
action_dim: 16
arms:
  left:  {action_start: 0,  action_end: 8,  driven: false}   # idle arm: hold, do NOT track
  right: {action_start: 8,  action_end: 16, driven: true}
action_type: absolute                 # absolute | delta_obs_t | delta_sequential
task_description: "Flip the package upside down."   # verbatim, conditioning input
chunk_size: 50
n_action_steps: 50
stats_floor: 0.03                     # if the dataset stats were floored
inference_config: configs/lerobot_control/inference_flip.yaml   # last known-good config
```

`cameras` is the highest-risk field — sending an *extra* camera is ignored by the model,
omitting or misspelling one is the fatal direction. `arms[].driven` matters because a
bimanual checkpoint trained on a task where one arm never moves emits "hold position" for
that arm, **not** a target to track; tracking it drives the idle arm toward a statistical
mean pose. `inference_config` is the cheapest single field that makes a checkpoint
re-deployable.

## Test with Fake Hardware First (Recommended)

```bash
# 1. Verify DDS connectivity + camera FPS (no model, no GPU needed)
./scripts/run_inference.sh --fake-hardware --monitor-enable up --build

# 2. Validate full pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --fake-hardware --profile inference up --build
```

> **Fake-hardware note:** `--echo-topic-only` / `ECHO_TOPIC_ONLY` and `MONITOR_ENABLE` env vars are
> **not** read by `docker-compose.fake-hardware.yml`. The monitor service hardcodes
> `echo_topic_only:=true` regardless; the inference service does not expose `MONITOR_ENABLE`.
> These variables only take effect with the production `docker-compose.yml`.

If `Control Loop` hits 30 Hz, the setup is ready for real hardware.

## Production (Real Robot)

```bash
# Standard inference
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh up --build

# With inference monitor
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --monitor-enable up --build

# Verify DDS connectivity without a checkpoint
./scripts/run_inference.sh --echo-topic-only up --build
```

> **`MODEL_PATH` must be absolute or start with `./`.** Bare relative paths are treated as named Docker volumes.
> ```bash
> MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last   # recommended
> MODEL_PATH=./model_zoo/my-task/checkpoints/last        # also valid
> ```

## Inference Config (`configs/lerobot_control/inference_default.yaml`)

Before running, review this file:

**Model**
```yaml
model:
  task_description: null
  # VLA-only (SmolVLA / Pi0 / Pi0.5): task prompt the model was trained on.
  # null = auto-read from anvil_config.json in the checkpoint (recommended).
```

**Per-model inference tuning** — override checkpoint defaults without retraining:
```yaml
inference_tuning:

  act:
    n_action_steps: null
    # Steps to execute per chunk before re-running inference.
    # null = use training value. Jittery? → raise. Hesitates? → lower.
    temporal_ensemble_coeff: null
    # Re-infers every step with exponentially weighted overlapping predictions.
    # Use 0.01 (paper default). Forces n_action_steps=1.

  diffusion:
    n_action_steps: null
    # Steps to execute per chunk. null = use training value.
    num_inference_steps: 10
    # Denoising iterations at inference time.
    # null = num_train_timesteps (100 steps, ~300ms on GPU).
    # 10   = ~30ms on GPU — recommended for real-time deployment.

  rtc:
    # VLA models only (SmolVLA / Pi0 / Pi0.5)
    inference_delay: 10
    # Fallback step-count before LatencyTracker auto-calibrates.
    # Rule of thumb: ceil(first_inference_ms × control_freq / 1000)
    queue_trigger_threshold: 50
    # Re-trigger inference when ActionQueue depth ≤ this.
    execution_horizon: 12
    # Steps consumed per chunk before the next inference fires.
    max_guidance_weight: 10.0
    prefix_attention_schedule: EXP
```

### State pinning (degenerate state dims)

Needed whenever one arm is idle for the whole task. Freezes that arm's state dims at the
training median instead of feeding live encoder values:

```yaml
state_pinning:
  enabled: true
  arms: [l]        # arm_mapping keys to pin
  source: q50      # q50 | mean, read from the checkpoint's own normalizer stats
  # values: [...]  # optional explicit override, one per pinned dim
```

**Why.** A joint that never moves in training has a degenerate `q99 − q01`, which the dataset
stats floor widens to `0.03`. QUANTILES normalisation is
`2·(x − q01)/(q99 − q01) − 1` (`normalize_processor.py`, QUANTILES branch), so those dims run
at `2/0.03 ≈ 67` units/rad against 1.3–2.6 for an arm that actually moves — ~35× more
sensitive. A sub-degree difference between the training rest pose and the deployed one pushes
the dim outside normalised `[-1, 1]`, and **nothing in `normalize_processor.py` clamps or
clips** (verified: no `clamp`/`clip` call anywhere in the file).

**Why it's fatal for pi0.5 but not SmolVLA.** pi0.5 discretises the normalised state into 256
bins and splices it into the **text prompt** (`processor_pi05.py:75-81`):

```python
discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

Below `-1` that yields token **`-1`** — a literal `"-1"` in the prompt string, never produced
during training. Above `+1` it saturates to **`255`**. Either corrupts the prefix that
conditions the *entire* action chunk — **both arms, not just the pinned dims**. The step
order is deliberate and load-bearing: `processor_pi05.py:141` carries the note
*"NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep"*.

SmolVLA is the mirror image: its `TokenizerProcessorStep` runs *before*
`NormalizerProcessorStep` (`processor_smolvla.py:69-85`), and state reaches the model through
`nn.Linear` (`modeling_smolvla.py:693`). Out-of-range is a large float that degrades smoothly,
not a bad token. That is exactly why `smolvla-flip` behaved and `pi05-flip` did not, on the
same dataset.

**Observed symptom:** right gripper pinned at 0.048–0.053 instead of reaching −0.003.
Deployment pose left j2 = −0.194 rad normalises to −1.84 and tokenises to `-1`.

Pinning to `q50` puts every pinned dim at exactly normalised `0.0` (token 128), so the state
portion of the prompt is byte-identical every step. Values are read from the checkpoint's
`policy_preprocessor.json` → `normalizer_processor` safetensors, so they cannot drift from the
weights. Set `enabled: false` to feed live state again and A/B against SmolVLA.

This also closes a silent path: `_build_observation` defaulted any joint missing from
`/joint_states` to `0.0`. For left j4 (`q01 = 1.574`) that normalises to **−105.9**, no warning.

<a id="safety-limits"></a>
**Safety limits:**
```yaml
# safety:
#   max_position_delta: 0.1
#   # Hard limit on joint position change per control step (radians).
#   min_position_delta: 0.05
#   # Minimum cumulative change before publishing a new command.
#   # Holds the last command until threshold is crossed — useful for
#   # overcoming motor dead zones / friction. Default: disabled (null).
```

> ⚠ **`min_position_delta` mixes units.** `ActionLimiter` applies this one scalar
> element-wise to a vector that mixes radians (j1–j7) with the **PRISMATIC** gripper in metres
> (`action_limiter.py:199`: `np.abs(self._pending_delta) >= self.min_delta_threshold`). The
> gripper only travels 0–0.05 m in total, so the `0.05` shown above — or anything at that
> scale — masks the gripper permanently while the arm keeps moving: it reaches, then never
> closes. Use `null`, or ~0.002 if friction genuinely demands a deadband.

**Image resolution:** frames are stored 640×480 and the model resizes to 224×224 internally.
Send native resolution — do not pre-scale.

## DDS Middleware Selection

Both Fast DDS and CycloneDDS are supported. **CycloneDDS is the default** (faster in our tests).

> ⚠ **Both sides must use the same RMW** — mixing Fast DDS and CycloneDDS = zero discovery (no error, just silence).

| Deployment | `RMW_IMPLEMENTATION` | `CYCLONEDDS_URI` | anvil-loader `.env.config` |
|-----------|----------------------|------------------|---------------------------|
| **Single-PC · CycloneDDS** *(default)* | `rmw_cyclonedds_cpp` | `file://.../single_pc.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=127.0.0.1` |
| Single-PC · Fast DDS | `rmw_fastrtps_cpp` | *(ignored)* | `ENABLE_CYCLONEDDS=false` |
| Two-PC · CycloneDDS | `rmw_cyclonedds_cpp` | `file://.../two_pc_gpu.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=<gpu_pc_ip>` |

All CycloneDDS configs live in `configs/cyclonedds/`. The defaults in `docker-compose.yml` and `.env.example` target single-PC CycloneDDS — override in `.env` to switch modes.

## Deployment Topologies

### Single-PC — inference and workcell on the same machine

```
  Same machine
┌────────────────────────────────────────────────────────────┐
│  anvil-loader (ros2_control)       anvil-embodied-ai       │
│  joint_states (500 Hz)  ◄─────────  inference_node (30 Hz) │
│  cameras (4× 30 Hz)      CycloneDDS  action commands       │
│                           (host net)                       │
└────────────────────────────────────────────────────────────┘
```

Both sides use CycloneDDS on the host network — multicast handles peer discovery automatically. Set in anvil-loader's `.env.config`:
```
ENABLE_CYCLONEDDS=true
CYCLONEDDS_PEER_IP=127.0.0.1
```

### Two-PC — GPU PC separate from the robot PC

```
  Anvil Devbox (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  ros2_control               │    │                    │    │  lerobot_control            │
│  joint_states (500 Hz)      │◄───┤  Gigabit Switch    ├───►│  inference_node (30 Hz)     │
│  cameras (4× 30 Hz)         │    │                    │    │  action commands            │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

Set `CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/two_pc_gpu.xml` and configure peer IPs in both `two_pc_gpu.xml` and anvil-loader's `.env.config`. See the [full documentation](https://docs.anvil.bot/) for network setup.

---

[← Back to README](../README.md)
