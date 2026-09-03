#!/usr/bin/env python3
"""Compare the live observation.state against the distribution it was trained on.

Why: a policy that reaches for nothing, or drifts to a pose that looks like an
average of the dataset, is usually being fed a state vector it never saw during
training. The offsets that cause this are small in joint space and invisible on
a plot — but the model does not consume joint space, it consumes
(x - mean) / std, and these checkpoints have per-dimension std floored to 0.0075
on the near-static arm. A 0.013 rad offset there lands at -3.7 sigma when
training never exceeded +/-1.9. This prints both views so the difference is
obvious.

The state vector is assembled exactly the way the inference node does it
(strategies/multi_process.py:208-227): joint names are
f"{observation_prefix}{sep}{arm_key}{sep}{joint_id}" over sorted(arm_mapping)
and then model_joint_order — NOT controller order. Reading /joint_states
directly rather than /monitor/obs_state means this works whether or not the
node was started with MONITOR_ENABLE.

Usage (inside the inference container, where the checkpoint is mounted):
    python3 check_obs_distribution.py [--seconds 20]
      [--model /workspace/model] [--config /workspace/config.yaml]
"""

import argparse
import glob
import os
import sys
import time

import yaml
from safetensors import safe_open

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


def load_state_stats(model_dir):
    """observation.state stats out of the checkpoint's preprocessor normalizer."""
    pattern = os.path.join(model_dir, "pretrained_model",
                           "policy_preprocessor_step_*_normalizer_processor.safetensors")
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no preprocessor normalizer under {model_dir}/pretrained_model")
    stats = {}
    with safe_open(files[0], framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("observation.state."):
                stats[key.split(".")[-1]] = f.get_tensor(key).flatten().tolist()
    missing = {"min", "max", "mean", "std"} - set(stats)
    if missing:
        sys.exit(f"normalizer is missing {sorted(missing)} for observation.state")
    return stats, os.path.basename(files[0])


def joint_names_from_config(config_path):
    cfg = yaml.safe_load(open(config_path))
    jn = cfg.get("joint_names", {})
    prefix = jn.get("observation_prefix", "follower")
    sep = jn.get("separator", "_")
    arm_mapping = jn.get("arm_mapping", {"l": "left", "r": "right"})
    order = jn.get("model_joint_order", [])
    if not order:
        sys.exit(f"{config_path} has no joint_names.model_joint_order")
    names, labels = [], []
    for arm_key in sorted(arm_mapping):
        for joint_id in order:
            names.append(f"{prefix}{sep}{arm_key}{sep}{joint_id}")
            labels.append(f"{arm_key.upper()}_{joint_id}")
    return names, labels


class Sampler(Node):
    def __init__(self, names):
        super().__init__("obs_distribution_check")
        self.names = names
        self.samples = []
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)

    def on_js(self, msg):
        d = dict(zip(msg.name, msg.position))
        row = [d.get(n) for n in self.names]
        if all(v is not None for v in row):
            self.samples.append(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--model", default="/workspace/model")
    ap.add_argument("--config", default="/workspace/config.yaml")
    args = ap.parse_args()

    stats, statfile = load_state_stats(args.model)
    names, labels = joint_names_from_config(args.config)
    n = len(names)
    if len(stats["mean"]) != n:
        sys.exit(f"config builds {n} dims but the checkpoint stats have "
                 f"{len(stats['mean'])} — config and checkpoint do not match")

    print(f"checkpoint : {args.model}")
    print(f"stats      : {statfile}")
    print(f"sampling /joint_states for {args.seconds:.0f}s ...")

    rclpy.init()
    node = Sampler(names)
    start = time.time()
    while time.time() - start < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()

    if not node.samples:
        sys.exit("no complete /joint_states samples — is the loader running?")

    cols = list(zip(*node.samples))
    print(f"samples    : {len(node.samples)}\n")

    hdr = (f"{'dim':<18}{'live min':>10}{'live max':>10}"
           f"{'train min':>11}{'train max':>11}{'worst sigma':>13}  status")
    print(hdr)
    print("-" * len(hdr))

    out_of_range, extreme = [], []
    for i, label in enumerate(labels):
        lo, hi = min(cols[i]), max(cols[i])
        tmin, tmax = stats["min"][i], stats["max"][i]
        mean, std = stats["mean"][i], stats["std"][i]
        # Training's own spread in sigma, for context on what "far" means here.
        train_sigma = max(abs(tmin - mean), abs(tmax - mean)) / std if std else 0.0
        sigma = max(abs(lo - mean), abs(hi - mean)) / std if std else 0.0

        if lo < tmin or hi > tmax:
            status = "OUT"
            out_of_range.append(label)
            if sigma > train_sigma * 1.5:
                status = "OUT (far)"
                extreme.append((label, sigma, train_sigma))
        else:
            status = "ok"
        print(f"{label:<18}{lo:>10.4f}{hi:>10.4f}{tmin:>11.4f}{tmax:>11.4f}"
              f"{sigma:>13.2f}  {status}")

    print()
    if not out_of_range:
        print("All dimensions inside the training range — the state fed to the "
              "policy is not the problem.")
        return 0

    print(f"{len(out_of_range)} dimension(s) outside the training range: "
          f"{', '.join(out_of_range)}")
    for label, sigma, train_sigma in extreme:
        print(f"  {label}: {sigma:.2f} sigma, training never exceeded "
              f"{train_sigma:.2f} — the model has no reference for this input")
    return 1


if __name__ == "__main__":
    sys.exit(main())
