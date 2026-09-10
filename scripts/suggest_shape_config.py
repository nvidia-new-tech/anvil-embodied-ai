#!/usr/bin/env python3
"""Suggest which shape config in configs/lerobot_control/shapes/ fits a checkpoint.

Reads the checkpoint's own config.json — state width and declared camera keys —
and reports every shape config whose contract matches. Does not guess: if more
than one matches, it says what you have to decide.

Usage:
    ./.venv/bin/python scripts/suggest_shape_config.py <checkpoint_dir>

Then confirm the pick with the real check:
    ./.venv/bin/python scripts/preflight_checkpoint.py <ckpt> --config <picked>
"""

import json
import sys
from pathlib import Path

import yaml

SHAPES = Path("configs/lerobot_control/shapes")


def resolve(ckpt: Path) -> Path:
    """Mirror ModelLoader's checkpoint resolution."""
    if (ckpt / "pretrained_model" / "config.json").exists():
        return ckpt / "pretrained_model"
    if not (ckpt / "config.json").exists():
        snaps = ckpt / "snapshots"
        if snaps.is_dir():
            for s in sorted(snaps.iterdir(), reverse=True):
                if (s / "config.json").exists():
                    return s
    return ckpt


def detect_static_dims(ckpt: Path, cfg: dict) -> list[int] | None:
    """Return state dims that never move in training, or None if undeterminable.

    A joint that is static has a degenerate spread that the dataset stats floor
    clamps to a fixed minimum. The signature is that the SAME minimum value
    repeats across several dims: independent joints never land on an identical
    spread to 4 decimal places by chance, so >=2 dims sharing the minimum means
    a floor was applied. A single small dim is just a gripper with short travel,
    not a floor.
    """
    pre = ckpt / "policy_preprocessor.json"
    if not pre.exists():
        return None
    state_file = None
    for step in json.loads(pre.read_text()).get("steps", []):
        if step.get("registry_name") == "normalizer_processor":
            state_file = step.get("state_file")
            break
    if not state_file or not (ckpt / state_file).exists():
        return None

    try:
        import numpy as np
        from safetensors.numpy import load_file
    except ImportError:
        return None

    stats = load_file(str(ckpt / state_file))
    mode = cfg.get("normalization_mapping", {}).get("STATE")
    if mode == "QUANTILES" and "observation.state.q01" in stats:
        spread = stats["observation.state.q99"] - stats["observation.state.q01"]
    elif "observation.state.std" in stats:
        spread = stats["observation.state.std"]
    else:
        return None

    lo = float(spread.min())
    if lo <= 0:
        return None
    at_floor = np.isclose(spread, lo, rtol=1e-4)
    # A floor is a repeated identical minimum. One lone small dim is a gripper.
    if at_floor.sum() < 2 or at_floor.all():
        return []
    return [int(i) for i in np.flatnonzero(at_floor)]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    ckpt = resolve(Path(sys.argv[1]))
    cfg_path = ckpt / "config.json"
    if not cfg_path.exists():
        print(f"ERROR  no config.json under {sys.argv[1]}")
        return 1

    cfg = json.loads(cfg_path.read_text())
    cams = sorted(
        k.rsplit(".", 1)[1]
        for k, v in cfg.get("input_features", {}).items()
        if v.get("type") == "VISUAL"
    )
    state = cfg["input_features"]["observation.state"]["shape"][0]
    action = cfg["output_features"]["action"]["shape"][0]

    anvil = {}
    if (ckpt / "anvil_config.json").exists():
        anvil = json.loads((ckpt / "anvil_config.json").read_text())

    print(f"Checkpoint  {ckpt}")
    print(f"  policy           : {cfg.get('type')}")
    print(f"  observation.state: {state}")
    print(f"  action           : {action}")
    print(f"  cameras          : {', '.join(cams)}")
    print(f"  normalization    : {cfg.get('normalization_mapping', {}).get('STATE')}")
    if anvil.get("task_description"):
        print(f"  task_description : '{anvil['task_description']}'")
    else:
        print("  task_description : MISSING — a VLA policy will refuse to start.")
        print("                     Set model.task_description in the config.")
    if state != action:
        print(f"  WARN  state {state} != action {action}; shape configs assume they match")

    matches = []
    for f in sorted(SHAPES.glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        jn = c["joint_names"]
        width = len(jn["arm_mapping"]) * len(jn["model_joint_order"])
        ccams = sorted(c["cameras"]["mapping"].values())
        if width == state and ccams == cams:
            matches.append((f, c))

    print()
    if not matches:
        print("No shape config matches. You need a new one — copy the closest and edit")
        print(f"arm_mapping and cameras.mapping to give state {state} and {cams}.")
        return 1

    for f, c in matches:
        pin = (c.get("state_pinning") or {}).get("enabled", False)
        pub = ", ".join(f"{a}{v['action_start']}:{v['action_end']}" for a, v in c["arms"].items())
        print(f"  MATCH  {f}")
        print(f"         publishes {pub}" + ("  + left-arm state pinning" if pin else ""))

    pick = matches[0][0]

    if len(matches) > 1:
        static = detect_static_dims(ckpt, cfg)
        n_joints = len(matches[0][1]["joint_names"]["model_joint_order"])
        left = set(range(n_joints))
        print()
        if static is None:
            print("  Cannot read normalizer stats, so this one is yours to decide:")
            print("    Does the left arm actually MOVE in this task?")
            print("      yes -> the plain variant      no -> the _rightonly variant")
        elif static and set(static) <= left:
            print(f"  Dims {static} sit at the stats floor, all inside the left arm")
            print(f"  {sorted(left)} -> that arm is STATIC in training.")
            print("  action[0:8] was trained to reproduce the current observation")
            print("  ('hold position'), not as a target to track. Use _rightonly.")
            for f, c in matches:
                if (c.get("state_pinning") or {}).get("enabled"):
                    pick = f
        elif static:
            print(f"  Dims {static} sit at the stats floor but are NOT confined to the")
            print(f"  left arm {sorted(left)}. Unexpected — inspect the stats yourself")
            print("  before choosing; neither variant is obviously right.")
        else:
            print("  No dim sits at a stats floor -> both arms move in training.")
            print("  Use the plain variant, which publishes both.")
            for f, c in matches:
                if not (c.get("state_pinning") or {}).get("enabled"):
                    pick = f

    print()
    print("Confirm with:")
    print(f"  ./.venv/bin/python scripts/preflight_checkpoint.py {sys.argv[1]} \\")
    print(f"    --config {pick}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
