#!/usr/bin/env python3
"""Run a real forward pass on a checkpoint with synthetic inputs — no robot, no ROS.

Preflight compares metadata. This actually loads the weights, builds a batch the
same way the ROS node does, and calls the policy. It catches what static checks
cannot: a processor pipeline that fails to load, a tokenizer missing from the
cache, VRAM that does not fit, real latency, and dead or saturated outputs.

Checks performed:
  1. Policy + processor pipelines load from the checkpoint
  2. A batch assembled per the config's arm_mapping / cameras survives preprocess
  3. select_action() returns finite actions of the declared width
  4. Actions land inside the training action range (from the unnormalizer stats)
  5. The policy REACTS to its input — different states must give different actions
  6. Latency and peak VRAM

Usage:
    ./.venv/bin/python scripts/offline_inference_test.py <ckpt> --config <yaml>
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def resolve(ckpt: Path) -> Path:
    if (ckpt / "pretrained_model" / "config.json").exists():
        return ckpt / "pretrained_model"
    if not (ckpt / "config.json").exists():
        snaps = ckpt / "snapshots"
        if snaps.is_dir():
            for s in sorted(snaps.iterdir(), reverse=True):
                if (s / "config.json").exists():
                    return s
    return ckpt


def build_batch(cfg, ycfg, state, device):
    """Mirror multi_process.py:196-227 — images [0,1] float CHW, state from arm_mapping."""
    batch = {}
    for key, feat in cfg["input_features"].items():
        if feat.get("type") == "VISUAL":
            c, h, w = feat["shape"]
            # mid-grey rather than zeros: a black frame is itself an edge case
            img = torch.full((1, c, h, w), 0.5, dtype=torch.float32)
            batch[key] = img.to(device)
    batch["observation.state"] = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    return batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=5, help="timed iterations")
    args = ap.parse_args()

    ckpt = resolve(Path(args.checkpoint))
    cfg = json.loads((ckpt / "config.json").read_text())
    ycfg = yaml.safe_load(Path(args.config).read_text())
    mtype = cfg["type"]
    state_w = cfg["input_features"]["observation.state"]["shape"][0]
    action_w = cfg["output_features"]["action"]["shape"][0]

    task = ""
    if (ckpt / "anvil_config.json").exists():
        task = json.loads((ckpt / "anvil_config.json").read_text()).get("task_description", "")
    task = (ycfg.get("model") or {}).get("task_description") or task

    is_vla = mtype in {"smolvla", "pi0", "pi05"}
    device = args.device if torch.cuda.is_available() else "cpu"

    if is_vla and not task:
        print(f"\nERROR  {mtype} needs a task_description and neither the checkpoint's")
        print("       anvil_config.json nor the config YAML provides one.")
        print("       The ROS node refuses to start on this (inference_node.py:382).")
        print("       Shape configs inherit it from the checkpoint, so for a checkpoint")
        print("       trained without one you must set it explicitly:")
        print("")
        print("         model:")
        print('           task_description: "..."')
        print("")
        print("       It must match the training string verbatim.")
        return 1
    print(f"Checkpoint  {ckpt}")
    print(f"  type {mtype} | state {state_w} | action {action_w} | device {device}")
    print(f"  task '{task}'")


    # --- 1. load -------------------------------------------------------------
    print("\n[1] Loading policy and processors")
    t0 = time.perf_counter()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor import PolicyProcessorPipeline

    # Load through the project's own ModelLoader, not from_pretrained directly.
    # ModelLoader has a low-memory path (build on the meta device, then
    # load_state_dict(assign=True)) that holds exactly ONE copy of the weights.
    # The stock loader allocates the architecture and then loads on top of it,
    # peaking at roughly twice the weight size — for pi0.5 that is ~22 GB and
    # the process is OOM-killed on a 30 GB host. Using ModelLoader also makes
    # this test exercise the same path the node does, which is the point.
    sys.path.insert(0, str(Path("ros2/src/lerobot_control").resolve()))
    try:
        from lerobot_control.model_loader import ModelLoader
    except ImportError:
        ModelLoader = None

    if ModelLoader is not None:
        class _Log:
            def info(self, m): print(f"  {m}")
            def warn(self, m): print(f"  WARN  {m}")
            def warning(self, m): print(f"  WARN  {m}")
            def error(self, m): print(f"  ERROR {m}")
            def debug(self, m): pass

        loader = ModelLoader(model_path=str(ckpt), device=device,
                             model_type=mtype, logger=_Log())
        policy, pre, post = loader.load_with_processors()
        policy.to(device).eval()
        print(f"  ok  loaded in {time.perf_counter()-t0:.1f}s via ModelLoader")
        return _run(args, cfg, ycfg, ckpt, mtype, state_w, action_w, task,
                    device, is_vla, policy, pre, post, t0)

    if mtype == "smolvla":
        import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as P
    elif mtype == "pi05":
        import lerobot.policies.pi05.processor_pi05  # noqa: F401
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P
    elif mtype == "pi0":
        import lerobot.policies.pi0.processor_pi0  # noqa: F401
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy as P
    elif mtype == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as P
    elif mtype == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as P
    else:
        print(f"  ERROR unsupported policy type {mtype}")
        return 1

    pcfg = PreTrainedConfig.from_pretrained(str(ckpt))
    pcfg.device = device
    policy = P.from_pretrained(str(ckpt), config=pcfg)
    policy.to(device).eval()

    pre = PolicyProcessorPipeline.from_pretrained(str(ckpt), config_filename="policy_preprocessor.json")
    post = PolicyProcessorPipeline.from_pretrained(str(ckpt), config_filename="policy_postprocessor.json")
    for p in (pre, post):
        if hasattr(p, "to"):
            try:
                p.to(device)
            except (TypeError, AttributeError):
                pass
    print(f"  ok  loaded in {time.perf_counter()-t0:.1f}s")
    return _run(args, cfg, ycfg, ckpt, mtype, state_w, action_w, task,
                device, is_vla, policy, pre, post, t0)


def _run(args, cfg, ycfg, ckpt, mtype, state_w, action_w, task,
         device, is_vla, policy, pre, post, t0):
    errors = []

    # training action range, for the sanity band
    from safetensors.numpy import load_file
    pj = json.loads((ckpt / "policy_postprocessor.json").read_text())
    sf = next(s["state_file"] for s in pj["steps"] if s["registry_name"] == "unnormalizer_processor")
    st = load_file(str(ckpt / sf))
    a_min, a_max = st["action.min"], st["action.max"]

    # --- 2/3. forward --------------------------------------------------------
    print("\n[2] Forward pass with synthetic inputs")
    rng = np.random.default_rng(0)
    mid = (a_min[:state_w] + a_max[:state_w]) / 2.0

    outs, lat = [], []
    for i in range(args.steps):
        # vary state within the training range so step 5 has something to compare
        frac = i / max(args.steps - 1, 1)
        state = a_min[:state_w] + (a_max[:state_w] - a_min[:state_w]) * (0.25 + 0.5 * frac)
        batch = build_batch(cfg, ycfg, state, device)
        if task:
            batch["task"] = [task]
        if hasattr(policy, "reset"):
            policy.reset()
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()

        # Mirrors inference_node.py exactly:
        #   preprocessor(batch) -> predict_action_chunk / select_action
        #   -> postprocessor.process_action(raw.squeeze(0))
        # No torch.inference_mode(): RTCProcessor calls torch.enable_grad()
        # internally for guidance gradients and inference_mode cannot be
        # overridden, which would silently zero them.
        obs = pre(batch)
        if is_vla:
            raw = policy.predict_action_chunk(obs)
        else:
            raw = policy.select_action(obs)
        act = post.process_action(raw.squeeze(0))

        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)

        arr = act.detach().float().cpu().numpy()
        chunk = arr.reshape(-1, action_w) if arr.ndim > 1 else arr.reshape(1, -1)
        a = chunk[0]                       # first step of the chunk
        outs.append(a)
        if i == 0:
            print(f"  raw {tuple(arr.shape)} -> chunk {chunk.shape[0]} x {chunk.shape[1]}")
            declared = cfg.get("n_action_steps")
            if is_vla and declared and chunk.shape[0] != declared:
                print(f"  note: chunk is {chunk.shape[0]}, config declares n_action_steps {declared}")
            if chunk.shape[1] != action_w:
                errors.append(f"action width {chunk.shape[1]} != declared {action_w}")
            if not np.isfinite(arr).all():
                errors.append("non-finite values in action chunk")

    outs = np.array(outs)
    print(f"  ok  {args.steps} steps, all finite: {bool(np.isfinite(outs).all())}")

    # --- 4. in-range ---------------------------------------------------------
    print("\n[3] Actions vs training range")
    lo, hi = a_min[:action_w], a_max[:action_w]
    span = hi - lo
    # Tolerance is 25% of the training span, but never tighter than an absolute
    # floor: a static joint's span can be ~1e-4, where any relative band is
    # meaningless and every output looks "out of range".
    tol = np.maximum(0.25 * span, 0.01)
    below, above = outs < (lo - tol), outs > (hi + tol)
    n_out = int((below | above).sum())
    degenerate = span < 0.05
    print(f"  outside train range (25% of span, min +/-0.01): {n_out}/{outs.size}")
    if degenerate.any():
        print(f"  note: dims {list(np.flatnonzero(degenerate))} have a near-zero training")
        print("        span — static joints; the absolute floor governs them.")
    for d in range(action_w):
        mark = "  <-- OUT" if (below[:, d] | above[:, d]).any() else ""
        flat = "  (static)" if degenerate[d] else ""
        print(f"    dim {d:2d}  {outs[:,d].min():9.4f}..{outs[:,d].max():9.4f}"
              f"   train {lo[d]:8.4f}..{hi[d]:8.4f}{flat}{mark}")
    if n_out:
        errors.append(f"{n_out} action values far outside the training range")

    # --- 5. responsiveness ---------------------------------------------------
    print("\n[4] Does the policy react to its input?")
    var = outs.std(axis=0)
    dead = int((var < 1e-6).sum())
    print(f"  per-dim std across {args.steps} different states: "
          f"min {var.min():.2e}  max {var.max():.2e}")
    print(f"  dims that never changed: {dead}/{action_w}")
    if dead == action_w:
        errors.append("output identical for every input — policy is ignoring observations")
    elif dead:
        print(f"  note: {dead} constant dim(s); expected for a static joint, "
              "suspicious otherwise")

    # --- 6. resources --------------------------------------------------------
    print("\n[5] Latency and memory")
    lat = np.array(lat)
    print(f"  first call {lat[0]:7.1f} ms  (includes warmup)")
    if len(lat) > 1:
        print(f"  steady     {lat[1:].mean():7.1f} ms  (min {lat[1:].min():.1f}, max {lat[1:].max():.1f})")
        hz = 1000 / lat[1:].mean()
        print(f"  -> {hz:.1f} Hz per chunk of {cfg.get('n_action_steps', '?')} steps")
    if device == "cuda":
        print(f"  peak VRAM  {torch.cuda.max_memory_allocated()/2**30:.2f} GiB "
              f"of {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    print()
    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  ERROR  {e}")
        return 1
    print("Forward pass OK — weights, processors and tokenizer all work offline.")
    print("Not covered: real camera timing, DDS, and whether the policy does the task.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
