#!/usr/bin/env python3
"""Separate PD tracking error from policy bias, using a monitor CSV.

A joint can sit away from where the policy asked for two very different reasons:

  * the PD loop in anvil-loader cannot hold it there (soft gain, gravity sag)
  * the policy is genuinely commanding a different pose

They look identical in obs_state alone. The discriminator is STEADY STATE: pick
the windows where the command is barely moving, so dynamic lag is excluded, and
look at what is left. A residual that correlates with the joint's PD gain is the
controller, not the policy.

Gains come from anvil-loader config/openarm_v2_inference.yaml (teleop_kp), where
j1-j4 are 40 while the wrist j5-j7 are 6-7 and the gripper is 0.1.

Usage:
    ./.venv/bin/python scripts/analyze_tracking_error.py monitor_output/inference_data.csv
"""

import argparse
import csv
import sys

import numpy as np

# teleop_kp from anvil-loader, controller order (j1..j7, finger)
DEFAULT_KP = [40.0, 40.0, 40.0, 40.0, 6.0, 7.0, 7.0, 0.1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--window", type=int, default=15,
                    help="steps the command must stay still to count as steady")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max command spread within the window (rad)")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(l for l in open(args.csv) if not l.startswith("#"))]
    if not rows:
        print(f"{args.csv} has no data rows — the monitor truncates this file on every "
              f"start, so copy it aside after each run.")
        return 1

    n = sum(1 for k in rows[0] if k.startswith("obs_state_"))
    obs = np.array([[float(r[f"obs_state_{i}"]) for i in range(n)] for r in rows])
    cmd = np.array([[float(r[f"control_cmd_{i}"]) for i in range(n)] for r in rows])
    kp = DEFAULT_KP[:n]
    names = [f"j{i+1}" for i in range(n - 1)] + ["finger"]
    arm = n - 1

    # Steady is judged PER JOINT. Requiring every joint to be still at once finds
    # nothing on a real run: the policy moves all of them continuously (measured
    # spread over a 5-step window is already 0.035-0.064 rad). Each joint has its
    # own quiet stretches, and that is all this needs.
    w = args.window
    steady = np.zeros((len(cmd), n), bool)
    for i in range(len(cmd) - w):
        seg = cmd[i:i + w]
        still = (seg.max(axis=0) - seg.min(axis=0)) < args.tol
        steady[i + w // 2] = still

    per = steady.sum(axis=0)
    print(f"{len(rows)} steps; steady windows per joint: "
          + ", ".join(f"{names[i]} {per[i]}" for i in range(n)))
    if per[:arm].max() < 30:
        print("too few steady windows — raise --tol or record a longer run")
        return 1

    err = cmd - obs
    print(f"\n{'':7} {'kp':>5} {'n':>5} {'residual mean':>14} {'std':>9} {'|mean|/std':>11}  reading")
    means = []
    for i in range(n):
        if per[i] < 30:
            print(f"{names[i]:7} {kp[i]:5} {per[i]:5}  (too few steady windows)")
            means.append(np.nan)
            continue
        e = err[steady[:, i], i]
        m, s = e.mean(), e.std()
        ratio = abs(m) / s if s > 0 else 0.0
        if ratio > 1.0:
            v = "systematic  << controller, not policy"
        elif ratio > 0.4:
            v = "leaning"
        else:
            v = "no systematic offset"
        means.append(abs(m))
        print(f"{names[i]:7} {kp[i]:5} {per[i]:5} {m:14.5f} {s:9.5f} {ratio:11.2f}  {v}")

    stiff = [i for i in range(arm) if kp[i] >= 20]
    soft = [i for i in range(arm) if kp[i] < 20]
    ms = np.nanmean([means[i] for i in stiff]) if stiff else 0
    mf = np.nanmean([means[i] for i in soft]) if soft else 0
    print(f"\nstiff joints {[names[i] for i in stiff]} (kp>=20): mean |residual| {ms:.5f} rad")
    print(f"soft  joints {[names[i] for i in soft]} (kp<20) : mean |residual| {mf:.5f} rad")
    if mf > 2 * ms:
        print("\n=> soft joints lag far more than stiff ones. That is the PD loop, not the\n"
              "   policy. Raise teleop_kp for those joints in anvil-loader before\n"
              "   changing anything on the inference side.")
    elif ms > 0 and mf < 1.5 * ms:
        print("\n=> residual does not track the gains, so the PD loop holds what it is\n"
              "   given. A pose offset here is the policy or the start pose.")
    else:
        print("\n=> inconclusive; collect a longer run with more steady time.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
