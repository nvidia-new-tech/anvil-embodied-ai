#!/usr/bin/env python3
"""Check a checkpoint against the inference YAML that will be used to run it.

Why: every way these two can disagree fails SILENTLY on the robot.

  - A camera the checkpoint declares but the YAML omits is substituted with a
    blank -1 image, attention mask 0 (modeling_pi05.py:1195-1204). No error, no
    warning; the policy just looks at a grey frame and decays toward a constant
    prior pose.
  - A camera the YAML lists but the checkpoint does not declare makes
    ObservationManager.has_complete_observation() wait forever for a feature
    with no slot. Inference never starts and the log says nothing useful.
  - A state width mismatch is absorbed by pi0/pi05 padding to max_state_dim
    (32) and truncating the action back down, so an 8-DOF checkpoint fed a
    16-DOF vector reads left-arm values in right-arm slots and still produces
    plausible-looking motion.
  - An arms.action_start/action_end slice past the real action width silently
    publishes zeros for the overhang.
  - task_description is a conditioning input; a reworded string degrades
    actions and reports nothing.

So compare them on the bench, before the arm is powered.

Usage:
    uv run scripts/preflight_checkpoint.py <checkpoint_dir> \
        --config configs/lerobot_control/inference_singlearm_flip.yaml

    # weights only, no YAML to compare against:
    uv run scripts/preflight_checkpoint.py <checkpoint_dir>

Exit status is 0 only when nothing is wrong; 1 if any ERROR was printed.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)
    print(f"  \033[31mERROR\033[0m  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  \033[33mWARN \033[0m  {msg}")


def ok(msg: str) -> None:
    print(f"  \033[32mok   \033[0m  {msg}")


def find_pretrained(root: Path) -> Path:
    """Locate the dir holding config.json — mirrors ModelLoader's auto-detection.

    Accepts the checkpoint dir itself, a dir containing pretrained_model/, or an
    HF cache layout with snapshots/<hash>/.
    """
    if (root / "config.json").exists():
        return root
    if (root / "pretrained_model" / "config.json").exists():
        return root / "pretrained_model"
    snapshots = root / "snapshots"
    if snapshots.is_dir():
        for snap in sorted(snapshots.iterdir()):
            if (snap / "config.json").exists():
                return snap
    raise SystemExit(f"No config.json found under {root} (looked in ./, ./pretrained_model, ./snapshots/*)")


# safetensors dtype -> numpy dtype. BF16 has no numpy equivalent, so it is read
# as uint16 and widened to float32 by shifting into the high half — bit-exact,
# since bf16 is just the top 16 bits of an f32.
_ST_DTYPES = {"F64": "<f8", "F32": "<f4", "F16": "<f2"}


def _scan_file_numpy(path: Path) -> tuple[int, list[str]]:
    """Read a .safetensors file with numpy alone and return (n_tensors, bad_keys).

    The format is: 8-byte little-endian header length, that many bytes of JSON
    describing each tensor's dtype/shape/data_offsets, then the raw buffer. No
    torch or safetensors package needed, so this runs on the host as well as
    inside the inference container.
    """
    import numpy as np

    bad: list[str] = []
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
        data_start = 8 + header_len
        buf = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start)

        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        for key, meta in tensors.items():
            dtype = meta.get("dtype")
            start, end = meta.get("data_offsets", (0, 0))
            raw = buf[start:end]
            if dtype == "BF16":
                u16 = raw.view("<u2").astype(np.uint32)
                arr = (u16 << 16).view(np.float32)
            elif dtype in _ST_DTYPES:
                arr = raw.view(_ST_DTYPES[dtype])
            else:
                continue  # integer / bool tensors cannot be NaN
            if not np.isfinite(arr).all():
                bad.append(f"{path.name}:{key}")
    return len(tensors), bad


def scan_safetensors(pretrained: Path) -> None:
    """Flag NaN/Inf weights.

    Not hypothetical: model_zoo/pi0.5-flip/002000 shipped with NaN in 466 of its
    813 tensors while the 005000 sibling was clean (noted in .env). A NaN
    checkpoint loads without complaint and emits NaN actions, which the action
    limiter then clamps into something that looks like a stuck arm.
    """
    files = sorted(pretrained.glob("*.safetensors"))
    if not files:
        warn("no .safetensors files found — skipped NaN/Inf scan")
        return

    try:
        import numpy  # noqa: F401
    except ImportError:
        warn("numpy not importable — skipped NaN/Inf scan")
        return

    total = 0
    bad: list[str] = []
    for f in files:
        n, b = _scan_file_numpy(f)
        total += n
        bad += b

    for key in bad[:5]:
        err(f"NaN/Inf in tensor {key}")
    if bad:
        err(f"{len(bad)}/{total} tensors contain NaN or Inf — do not deploy this checkpoint")
    else:
        ok(f"{total} tensors scanned across {len(files)} file(s), 0 NaN / 0 Inf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path, help="checkpoint dir (or its pretrained_model/)")
    ap.add_argument("--config", type=Path, default=None, help="inference YAML to cross-check")
    ap.add_argument("--skip-weight-scan", action="store_true", help="skip the NaN/Inf tensor scan (it reads every weight)")
    args = ap.parse_args()

    pretrained = find_pretrained(args.checkpoint)
    cfg = json.loads((pretrained / "config.json").read_text())

    anvil_path = pretrained / "anvil_config.json"
    anvil = json.loads(anvil_path.read_text()) if anvil_path.exists() else {}

    inputs = cfg.get("input_features", {})
    outputs = cfg.get("output_features", {})
    ckpt_cams = sorted(k.split("observation.images.")[-1] for k in inputs if k.startswith("observation.images."))
    state_shape = inputs.get("observation.state", {}).get("shape", [])
    action_shape = outputs.get("action", {}).get("shape", [])
    state_dim = state_shape[0] if state_shape else None
    action_dim = action_shape[0] if action_shape else None

    print(f"\n\033[1mCheckpoint\033[0m  {pretrained}")
    print(f"  policy type      : {cfg.get('type')}")
    print(f"  cameras          : {', '.join(ckpt_cams) or '(none)'}")
    print(f"  observation.state: {state_dim}")
    print(f"  action           : {action_dim}")
    print(f"  chunk_size       : {cfg.get('chunk_size')}   n_action_steps: {cfg.get('n_action_steps')}")
    print(f"  max_state_dim    : {cfg.get('max_state_dim')}   max_action_dim: {cfg.get('max_action_dim')}")
    if anvil:
        print(f"  action_type      : {anvil.get('action_type')}")
        print(f"  task_description : {anvil.get('task_description')!r}")
        if anvil.get("note"):
            print(f"  note             : {anvil['note']}")
    else:
        print("  anvil_config.json: MISSING")

    print("\n\033[1mWeights\033[0m")
    if args.skip_weight_scan:
        print("  (skipped)")
    else:
        scan_safetensors(pretrained)

    if not anvil:
        warn("no anvil_config.json — action_type falls back to the launch default (absolute); confirm it matches training")

    if not args.config:
        print("\nNo --config given; skipped YAML cross-check.")
        return finish()

    if yaml is None:
        err("PyYAML not importable — cannot cross-check the config")
        return finish()

    print(f"\n\033[1mConfig\033[0m  {args.config}")
    y = yaml.safe_load(args.config.read_text())

    # ---- cameras -------------------------------------------------------------
    mapping = (y.get("cameras") or {}).get("mapping") or {}
    yaml_cams = sorted(mapping.values())
    if yaml_cams == ckpt_cams:
        ok(f"cameras match: {', '.join(yaml_cams)}")
    else:
        missing = [c for c in ckpt_cams if c not in yaml_cams]
        extra = [c for c in yaml_cams if c not in ckpt_cams]
        if missing:
            err(f"checkpoint declares {missing} but the config never sends them — "
                f"the policy will receive a blank -1 image for each, silently")
        if extra:
            err(f"config sends {extra} which the checkpoint does not declare — "
                f"has_complete_observation() will block forever and inference will never start")
        print(f"         checkpoint: {ckpt_cams}")
        print(f"         config    : {yaml_cams}")

    # ---- state width ---------------------------------------------------------
    jn = y.get("joint_names") or {}
    arm_mapping = jn.get("arm_mapping") or {}
    joint_order = jn.get("model_joint_order") or []
    state_features = jn.get("state_features") or ["position"]
    yaml_state_dim = len(arm_mapping) * len(joint_order)

    if state_dim is None:
        warn("checkpoint declares no observation.state — skipped state width check")
    elif yaml_state_dim == state_dim:
        ok(f"observation.state width {state_dim} "
           f"= {len(arm_mapping)} arm(s) {sorted(arm_mapping)} × {len(joint_order)} joints")
    else:
        err(f"state width mismatch: config builds {yaml_state_dim} "
            f"({len(arm_mapping)} arm(s) {sorted(arm_mapping)} × {len(joint_order)} joints) "
            f"but the checkpoint expects {state_dim}. pi0/pi05 pad to max_state_dim and will "
            f"NOT error — every dimension lands in the wrong slot")

    if len(state_features) != 1 or state_features[0] != "position":
        warn(f"state_features = {state_features}; the checkpoint's input_features declare "
             f"{[k for k in inputs if k.startswith('observation.') and 'images' not in k]}")

    # ---- action slices -------------------------------------------------------
    arms = y.get("arms") or {}
    if action_dim is None:
        warn("checkpoint declares no action feature — skipped slice check")
    else:
        covered: set[int] = set()
        for name, a in arms.items():
            s = a.get("action_start", 0)
            e = a.get("action_end", action_dim)
            if e > action_dim:
                err(f"arms.{name} slices action[{s}:{e}] but the action is only {action_dim} wide — "
                    f"the overhang publishes zeros")
            elif s >= e:
                err(f"arms.{name} has an empty slice action[{s}:{e}]")
            else:
                ok(f"arms.{name} → action[{s}:{e}] ({e - s} joints) on {a.get('command_topic')}")
            if e - s != len(joint_order):
                err(f"arms.{name} slice is {e - s} wide but controller_joint_order/model_joint_order "
                    f"has {len(joint_order)} joints")
            covered |= set(range(s, min(e, action_dim)))
        uncommanded = sorted(set(range(action_dim)) - covered)
        if uncommanded:
            warn(f"action dims {uncommanded} are never published (no arm claims them). "
                 f"Deliberate for a bimanual checkpoint whose idle arm must not be tracked "
                 f"(see docs/inference.md, arms[].driven) — unexpected otherwise")

    # ---- task description ----------------------------------------------------
    yaml_task = (y.get("model") or {}).get("task_description")
    ckpt_task = anvil.get("task_description")
    if ckpt_task is None:
        warn("no task_description in the checkpoint to compare against")
    elif yaml_task is None:
        ok(f"task_description null → auto-read from checkpoint: {ckpt_task!r}")
    elif yaml_task == ckpt_task:
        ok(f"task_description matches: {ckpt_task!r}")
    else:
        err(f"task_description differs — the language tokens are a conditioning input, "
            f"a mismatch degrades actions with no error.\n"
            f"         checkpoint: {ckpt_task!r}\n"
            f"         config    : {yaml_task!r}")

    # ---- RTC / chunk ---------------------------------------------------------
    rtc = (y.get("inference_tuning") or {}).get("rtc") or {}
    chunk = cfg.get("chunk_size")
    thresh = rtc.get("queue_trigger_threshold")
    if chunk and thresh and thresh > chunk:
        warn(f"queue_trigger_threshold ({thresh}) exceeds chunk_size ({chunk}) — "
             f"the queue can never hold that many, so inference re-triggers every step")

    # ---- safety --------------------------------------------------------------
    safety = y.get("safety") or {}
    mpd = safety.get("min_position_delta")
    if mpd is not None and mpd >= 0.01:
        err(f"min_position_delta = {mpd}. ActionLimiter applies one scalar threshold to a vector "
            f"mixing radians (joint1-7) with the PRISMATIC gripper in metres (0..0.05 m travel). "
            f"Anything at this scale freezes the gripper: it reaches, then never closes. Use null, "
            f"or ~0.002 if friction demands a deadband")

    return finish()


def finish() -> int:
    print()
    if ERRORS:
        print(f"\033[31m{len(ERRORS)} error(s)\033[0m, {len(WARNINGS)} warning(s) — fix the errors before launching.")
        return 1
    if WARNINGS:
        print(f"\033[33m{len(WARNINGS)} warning(s)\033[0m, no errors — review, then launch.")
        return 0
    print("\033[32mAll checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
