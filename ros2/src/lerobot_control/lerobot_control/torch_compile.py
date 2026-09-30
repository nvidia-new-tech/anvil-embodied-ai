"""Opt-in torch.compile of a VLA policy's denoise step, with warm-up before first use.

Enabled by the ANVIL_TORCH_COMPILE environment variable, whose value is the
torch.compile mode ("reduce-overhead", "default", "max-autotune-no-cudagraphs").
Unset or empty = off, which is the default everywhere.

Why: SmolVLA's RTC chunk is launch-bound, not compute-bound. One RTC call issues
~14k small CUDA kernels; the GPU is busy for only ~25-50% of the call and the rest
is Python/PyTorch dispatch on one CPU core. On Jetson Thor (ARM cores) that is
~245 ms per chunk against ~150 ms on an x86 RTX 5080 laptop. Compiling
denoise_step with mode="reduce-overhead" (fused kernels + CUDA graphs) measured
~70 ms on Thor, with no recompilation when inference_delay and the leftover length
vary from call to call (RTC guidance runs outside the compiled function).

Numerics: compiled vs eager differs by ~3e-3 in normalized action units under
bf16, the usual size of kernel-fusion rounding. That is a behaviour change, which
is why this is opt-in rather than on by default.

Cost: the first compile takes ~55 s on Thor. Point TORCHINDUCTOR_CACHE_DIR and
TRITON_CACHE_DIR at a persistent volume to reuse it across restarts.

Thor also needs TRITON_PTXAS_BLACKWELL_PATH set to a ptxas that knows sm_110
(torch's bundled CUDA 13 one): Triton 3.6's own ptxas-blackwell is CUDA 12.8 and
fails with "Value 'sm_110a' is not defined for option 'gpu-name'".
"""

from __future__ import annotations

import os
import time

import torch

ENV_VAR = "ANVIL_TORCH_COMPILE"
_MODES = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
_WARMUP_ROUNDS = 3


def requested_mode() -> str | None:
    """Return the requested torch.compile mode, or None when compile is off."""
    mode = os.environ.get(ENV_VAR, "").strip()
    if not mode or mode.lower() in {"0", "off", "none", "false"}:
        return None
    if mode not in _MODES:
        raise ValueError(f"{ENV_VAR}={mode!r} is not a torch.compile mode; expected one of {sorted(_MODES)}")
    return mode


def _synthetic_batch(policy, task: str, device: str) -> dict:
    """A batch shaped like the node's observations: grey images, zero state."""
    batch = {}
    for key, feat in policy.config.input_features.items():
        kind = getattr(getattr(feat, "type", None), "name", str(getattr(feat, "type", "")))
        if kind == "VISUAL":
            batch[key] = torch.full((1, *feat.shape), 0.5, dtype=torch.float32, device=device)
        elif kind == "STATE":
            batch[key] = torch.zeros((1, *feat.shape), dtype=torch.float32, device=device)
    if task:
        batch["task"] = [task]
    return batch


def maybe_compile_vla(policy, preprocessor, task: str, device: str, logger) -> bool:
    """Compile policy.model.denoise_step if ANVIL_TORCH_COMPILE asks for it, then warm up.

    Returns True when the compiled path is active. On any failure the eager
    denoise_step is restored and the error is logged at ERROR level, so a broken
    compile toolchain degrades to the old latency instead of taking inference down.
    """
    mode = requested_mode()
    if mode is None:
        return False
    inner = getattr(policy, "model", None)
    if inner is None or not hasattr(inner, "denoise_step"):
        logger.error(f"{ENV_VAR}={mode}: policy has no model.denoise_step; running eager")
        return False
    if not str(device).startswith("cuda"):
        logger.error(f"{ENV_VAR}={mode}: device is {device}, not CUDA; running eager")
        return False

    eager = inner.denoise_step
    logger.info(f"{ENV_VAR}={mode}: compiling denoise_step and warming up (first build ~1 min on Thor)...")
    t0 = time.monotonic()
    try:
        inner.denoise_step = torch.compile(eager, mode=mode)
        obs = preprocessor(_synthetic_batch(policy, task, device)) if preprocessor else _synthetic_batch(policy, task, device)
        rtc_cfg = getattr(policy.config, "rtc_config", None)
        eh = getattr(rtc_cfg, "execution_horizon", 10) if rtc_cfg is not None else 10
        # Warm both call shapes the inference thread uses: a first chunk without a
        # leftover, and RTC-guided chunks with one. Mirrors _inference_loop: no
        # inference_mode(), since RTC guidance needs autograd.
        chunk = policy.predict_action_chunk(obs)
        for _ in range(_WARMUP_ROUNDS):
            left = chunk.squeeze(0)[eh:].clone()
            chunk = policy.predict_action_chunk(obs, inference_delay=4, prev_chunk_left_over=left, execution_horizon=eh)
        torch.cuda.synchronize()
        if not torch.isfinite(chunk).all():
            raise RuntimeError("compiled warm-up produced non-finite actions")
    except Exception as e:  # noqa: BLE001 - any toolchain failure must fall back
        inner.denoise_step = eager
        logger.error(f"{ENV_VAR}={mode}: compile/warm-up failed, running EAGER instead: {type(e).__name__}: {e}")
        return False
    finally:
        if hasattr(policy, "reset"):
            policy.reset()

    logger.info(f"{ENV_VAR}={mode}: ready in {time.monotonic() - t0:.1f}s")
    return True
