---
name: run-inference
description: "Start, stop or switch Anvil inference — joint-space or EE-space, on the real robot or against fake hardware. Use when asked to run, launch, restart or stop inference, to switch between the joint-space and EE-space models, to change control frequency, or when a run produced no monitor CSV. Covers the launch preconditions that fail silently: the monitor compose profile, the loader-side subscriber, and the model/config pairing. Working directory ~/Desktop/anvil/anvil-embodied-ai."
---

# Running inference

## Use `run_inference.sh`, never `docker compose up`

```bash
./scripts/run_inference.sh --monitor-enable up
```

The monitor is a **separate service behind the `monitor` compose profile**
(`inference-monitor`). Only the script injects `--profile monitor`.
`MONITOR_ENABLE=true` on its own is just a launch argument — the run looks
completely normal and **writes no CSV at all**. The script also pre-creates the
output directory as the invoking user; skip that and the container leaves a
root-owned empty `monitor_output/inference_data.csv` behind, which then blocks
writes and needs sudo to remove.

Confirm it actually started:

```bash
docker ps --format '{{.Names}}' | grep monitor    # want lerobot-inference-monitor
```

## Switching modes — two variables, no file edits

Shell environment beats `.env` in compose substitution, and `run_inference.sh`
is written for this (it auto-detects `ACTION_TYPE` from `MODEL_PATH`). Passing
them inline also makes a run self-describing instead of depending on whatever
`.env` was last left at.

**Joint-space:**

```bash
cd ~/Desktop/anvil/anvil-embodied-ai
MODEL_PATH=$PWD/model_zoo/smolvla_pack_pick_place_20260917/010000/pretrained_model \
CONFIG_FILE=./configs/lerobot_control/shapes/1arm_2cam.yaml \
CONTROL_FREQ=30.0 ./scripts/run_inference.sh --monitor-enable up
```

**EE-space:**

```bash
cd ~/Desktop/anvil/anvil-embodied-ai
MODEL_PATH=$PWD/model_zoo/smolvla_pack_pick_place_eef_5000 \
CONFIG_FILE=./configs/lerobot_control/shapes/1arm_2cam_ee.yaml \
CONTROL_FREQ=30.0 ./scripts/run_inference.sh --monitor-enable up
```

Everything else — DDS settings, HF cache, `LEROBOT_EXTRAS`, `IMAGE_TAG` — is
shared and stays in `.env`. `IMAGE_TAG=ee-space` works for both: it is `latest`
plus `anvil_msgs`, and the joint path is unaffected.

**The model and the config must match.** `preflight_checkpoint.py` cannot tell
an EE checkpoint from a joint one — see the `checkpoint-intake` skill before
pairing a checkpoint you have not run before.

## Preconditions

1. **Loader up**, arm homed:
   ```bash
   cd ~/Desktop/anvil/anvil-loader && docker compose ps
   ```
   If it is down, starting it **auto-homes the arm — it will move**. Say so and
   get clearance before doing it.

2. **The command topic has a subscriber.** For EE-space this needs
   `commanded_ee: true` on the loader side
   (`ARMS_CONTROL_CONFIG_FILE=openarm_v2_inference_commanded_ee.yaml`). Without
   it `/commanded_ee_right` does not exist, inference publishes into the void,
   and the arm simply never moves with nothing in any log. Verify with the
   throwaway-container recipe in the `workcell-doctor` skill.

   The loader does **not** need switching back for joint-space:
   `commanded_ee: true` adds a subscriber rather than replacing the joint one,
   so both topics stay live. Leaving it avoids a restart and another homing.

3. **EE-space only:** `docker-compose.override.yml` must be present. The
   `:ee-space` image was built by committing `anvil_msgs` into `:latest`, so its
   `lerobot_control` is the pre-EE source; the override mounts the repo's copy
   over it. Without it the guarded import in `inference_node.py` sets
   `CommandedEEPose = None` and the EE arm is dropped from the publish path
   silently. Delete the file once the image is rebuilt from the merged
   Dockerfile.

## Control frequency

Default 30.0. For a first run of anything new, start low and climb:

```bash
CONTROL_FREQ=5.0 ...     # then 10 → 20 → 30
```

At 5 Hz the motion is inherently choppy — that is the frequency, not the policy.

⚠️ Raising `CONTROL_FREQ` does not change the per-step position cap but does
change how many steps happen per second. With `max_position_delta_m: 0.005`,
5 Hz is 25 mm/s and 30 Hz is 150 mm/s. Read the two together.

## Safety

- **Do not add `-d`** for a real run. Staying in the foreground makes `Ctrl-C`
  the software stop.
- Person at the e-stop, arms clear.
- **Joint-space has no limiting at all** — `safety.max_relative_target` and
  `safety.joint_limits` are both `null`, model output reaches the controller
  unmodified. EE-space is the stricter of the two.
- To validate something new without moving the arm, use the dry-run technique:
  point `command_topic` at an unsubscribed name and read the commands back.

## Stopping

```
1. stop inference (Ctrl-C, or ./scripts/run_inference.sh down)
2. dehome from the web UI (http://localhost:3000) — wait for the arm to lower
3. only then: cd ~/Desktop/anvil/anvil-loader && docker compose down
```

`compose down` does **not** lower the arm. Its `pre_stop.sh` only posts an
analytics event. The joint motors hold position by torque for as long as they
are powered, so cutting power while the arm is up drops it.

## After a run

`monitor_output/` gets a per-step CSV and an auto-plotted PNG. Watch the gripper
dimension — pinned at a constant is a config problem, not a model problem.
