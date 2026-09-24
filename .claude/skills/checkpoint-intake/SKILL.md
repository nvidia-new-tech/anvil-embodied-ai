---
name: checkpoint-intake
description: "Verify a new SmolVLA/ACT/pi0 checkpoint before it ever drives the Anvil arm — pick the matching shape config, scan the weights, check the contract, and run a real forward pass offline. Use whenever a new checkpoint lands in model_zoo/, when MODEL_PATH is about to change, when someone asks whether a checkpoint is safe or ready to run, or when a policy behaves oddly and the checkpoint/config pairing is suspect. Critically: preflight_checkpoint.py CANNOT distinguish an EE-space checkpoint from a joint-space one — both are 8 wide and it reports All checks passed either way — so always confirm the action space from the normalizer as described here."
---

# Checkpoint intake

Run from `~/Desktop/anvil/anvil-embodied-ai`. The full reference is
`docs/checkpoint驗收流程.md`; this is the working procedure.

Almost every way a checkpoint and its config can disagree fails **silently**.
Verify against the checkpoint's own metadata, never against a filename, a note,
or the last config that worked.

## 0. Point at the right directory

A training run produces `<step>/pretrained_model/`, not a flat directory:

```
model_zoo/smolvla_pack_pick_place_20260917/005000/pretrained_model/   ← this
model_zoo/smolvla_pack_pick_place_eef_5000/                           ← or flat, varies
```

`MODEL_PATH` must point at the directory containing `model.safetensors`.

## 1. Which shape config fits

```bash
./.venv/bin/python scripts/suggest_shape_config.py <ckpt>
```

Reports policy type, state/action widths, cameras, normalization, and
`task_description`. `task_description : MISSING` means the checkpoint was
trained without one — a VLA policy refuses to start, so set it verbatim in the
shape config (get the string from a sibling checkpoint of the same run).

## 2. ⚠️ Confirm the action space yourself

**`preflight_checkpoint.py` cannot tell EE-space from joint-space.** It compares
widths (8 == 8) and `config.json` carries no action names, so a joint-space
checkpoint against `1arm_2cam_ee.yaml` reports `All checks passed`. Running that
pairing publishes joint angles in radians as metres and a quaternion.

The gripper's physical range `[-0.003, 0.05]` metres is recognisable in both,
and **which dimension it lands in** is the tell:

```bash
./.venv/bin/python -c "
from safetensors.numpy import load_file
import numpy as np, glob, sys
d = sys.argv[1]
f = glob.glob(d + '/policy_postprocessor_step_*_unnormalizer_processor.safetensors')[0]
x = load_file(f)
mn = np.asarray(x['action.min']).ravel(); mx = np.asarray(x['action.max']).ravel()
print('dim 0:', f'{mn[0]:+.4f} .. {mx[0]:+.4f}')
print('dim 7:', f'{mn[7]:+.4f} .. {mx[7]:+.4f}')
" <ckpt>
```

- gripper range at **dim 0** → joint-space (`finger_joint1` leads
  `model_joint_order`) → `shapes/1arm_2cam.yaml`
- gripper range at **dim 7** → EE-space (`[x,y,z,qx,qy,qz,qw,gripper]`) →
  `shapes/1arm_2cam_ee.yaml`

Real values for comparison:

```
joint-space   dim 0: -0.0030 .. +0.0500     dim 7: -1.2149 .. +1.4004  (radians)
EE-space      dim 0: +0.1509 .. +0.5002     dim 7: -0.0011 .. +0.0517  (metres)
```

`anvil_config.json`'s `note` usually says which it is, but that is a
hand-written string; the normalizer is derived from the training data.

## 3. Preflight

```bash
./.venv/bin/python scripts/preflight_checkpoint.py <ckpt> \
  --config configs/lerobot_control/shapes/<picked>.yaml
```

Scans every tensor for NaN/Inf (this has caught real corruption, always from a
bad **copy**, not a bad export) and cross-checks camera keys, state width,
action slice and `task_description`. **Must exit 0.** If it does not, fix what
it reports — do not launch anyway.

## 4. Normalizer stats

```bash
./.venv/bin/python -c "
from safetensors.numpy import load_file
import numpy as np, glob, sys
f = glob.glob(sys.argv[1] + '/policy_preprocessor_step_*_normalizer_processor.safetensors')[0]
s = np.asarray(load_file(f)['observation.state.std']).ravel()
print('std:', np.array2string(s, precision=4, suppress_small=True))
u, c = np.unique(np.round(s, 6), return_counts=True)
rep = [(x, n) for x, n in zip(u, c) if n > 1]
print('repeated values (degenerate dims):', rep or 'none')
" <ckpt>
```

**Repeated identical std values** are the signature of a static joint clamped to
the dataset stats floor — independent joints never land on the same spread by
chance. That means high-gain noise channels; see state pinning in
`docs/checkpoint驗收流程.md`. A merely small std with no repeats is a
low-movement joint, which is fine.

For a bimanual checkpoint this also decides `_rightonly` vs the plain variant:
static left arm → `_rightonly`, because `action[0:8]` was trained to reproduce
the current observation, and tracking it makes the arm creep.

## 5. Offline forward pass

Run in the inference image, not the host venv — the host `.venv` has no
`transformers` for the VLA policies.

```bash
docker run --rm --gpus all -e HF_HUB_OFFLINE=1 -e HF_HOME=/hf \
  -v ~/.cache/huggingface:/hf:ro -v "$PWD":/workspace/repo:ro \
  -v "$PWD/model_zoo":/model_zoo:ro \
  ghcr.io/anvil-robotics/lerobot-inference:ee-space \
  python3 /workspace/repo/scripts/offline_inference_test.py /model_zoo/<ckpt> \
  --config /workspace/repo/configs/lerobot_control/shapes/<picked>.yaml
```

Catches what static checks cannot: processor pipelines that fail to construct, a
tokenizer missing from `HF_CACHE`, VRAM/host-RAM that does not fit, dead outputs
(identical action for every input), and values far outside the training range.

Reference numbers on this machine (RTX 5080 Laptop, 15.5 GiB): load 9-60 s,
steady 120-160 ms per 50-step chunk, peak VRAM ~0.90 GiB. pi0.5 does **not**
load here — it is killed during `from_pretrained` on host RAM, not VRAM.

## 6. Wire up and launch

`MODEL_PATH` and `CONFIG_FILE` must be the pair preflight approved. Prefer
passing them on the command line rather than editing `.env` — see the
`run-inference` skill.

First real run: `--monitor-enable`, and watch the gripper dimension. Pinned at a
constant is the classic symptom of a config problem, not a model problem.

## Reporting

Say which shape config matched and why, that the action space was confirmed from
the normalizer (not assumed), the preflight exit code, and the measured latency
and VRAM. If anything failed, say what and stop — do not suggest launching.
