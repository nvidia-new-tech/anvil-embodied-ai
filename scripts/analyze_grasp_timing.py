#!/usr/bin/env python3
"""Answer one question from a monitor CSV: was the arm in place when the gripper shut?

That is the measurement that separates the competing explanations for a grasp
that closes above the table:

  A. execution is too slow  -> arm is rate-limited and lags the model's target
  B. the target keeps moving -> model still commands descent AFTER the gripper
                                shut, so the arm never converges
  C. neither                 -> arm arrived; the shortfall is elsewhere
                                (calibration, no force feedback)

Usage:
    ./.venv/bin/python scripts/analyze_grasp_timing.py monitor_output/inference_data.csv

Column layout (inference_node._publish_monitor):
    obs_state / control_cmd / delta_cmd  -> CONTROLLER order (gripper LAST)
    raw_output                           -> MODEL order      (gripper FIRST)
"""

import argparse
import csv
import sys

import numpy as np


def load(path):
    rows = [r for r in csv.DictReader(l for l in open(path) if not l.startswith("#"))]
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    n = sum(1 for k in rows[0] if k.startswith("obs_state_"))

    def block(prefix):
        return np.array([[float(r[f"{prefix}_{i}"]) for i in range(n)] for r in rows])

    t = np.array([float(r["timestamp"]) for r in rows])
    return t - t[0], block("obs_state"), block("raw_output"), block("delta_cmd"), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--closed-below", type=float, default=0.02,
                    help="gripper position counted as closed (metres)")
    args = ap.parse_args()

    t, obs, raw, dcmd, n = load(args.csv)
    arm = n - 1                      # gripper is last in controller order
    tgt = np.concatenate([raw[:, 1:n], raw[:, 0:1]], axis=1)   # -> controller order
    err = np.abs(tgt[:, :arm] - obs[:, :arm]).max(axis=1)
    grip = obs[:, arm]

    cap = np.abs(dcmd[:, :arm]).max()
    sat = (np.abs(np.abs(dcmd[:, :arm]) - cap) < 1e-6).any(axis=1) if cap > 0 else np.zeros(len(t), bool)

    print(f"{len(t)} steps, {t[-1]:.1f}s, {n} joints")
    print(f"gripper travel {grip.min():.4f}..{grip.max():.4f} m")
    print(f"largest per-step command delta seen: {cap:.4f} "
          f"({'a cap appears active' if (np.abs(np.abs(dcmd[:, :arm]) - cap) < 1e-6).sum() > 3 else 'no cap evident'})")

    closed = grip < args.closed_below
    events = [i for i in range(1, len(closed)) if closed[i] and not closed[i - 1]]
    if not events:
        print(f"\nno gripper close below {args.closed_below} m — nothing to analyse")
        return 0

    print(f"\ngripper closed {len(events)}x at t = " + ", ".join(f"{t[i]:.2f}s" for i in events))
    verdicts = []
    for i in events:
        after = slice(i, min(i + 45, len(t)))
        still = tgt[after, :arm] - obs[after, :arm]
        # does the model keep asking for MORE motion after the gripper shut?
        growing = np.abs(still).max(axis=1)
        pre = slice(max(i - 12, 0), i)

        print(f"\n--- close at t={t[i]:.2f}s (step {i}) ---")
        print(f"  arm error when the gripper shut : {err[i]:.4f} rad")
        print(f"  rate-limited in the 12 steps before : {100 * sat[pre].mean():.0f}% of steps")
        print(f"  arm error over the next {len(growing)} steps : "
              f"{growing[0]:.4f} -> {growing.min():.4f} (min) -> {growing[-1]:.4f}")

        if err[i] < 0.02:
            v = "C: arm was in place — look at calibration / force feedback, not tuning"
        elif growing[-1] > growing[0]:
            v = "B: target keeps moving after close — raise execution_horizon, fix inference_delay"
        elif sat[pre].mean() > 0.5:
            v = "A: arm rate-limited and lagging — max_relative_target is throttling it"
        else:
            v = "B/C: arm lagged but was not rate-limited — target motion or dynamics"
        print(f"  => {v}")
        verdicts.append(v[0])

    print("\n" + "=" * 60)
    top = max(set(verdicts), key=verdicts.count)
    print(f"dominant verdict across {len(events)} grasp(s): {top}")
    print("compare this number between runs:  arm error when the gripper shut")
    return 0


if __name__ == "__main__":
    sys.exit(main())
