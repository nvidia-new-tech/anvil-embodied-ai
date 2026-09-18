"""Safety limiting for EE-space (CommandedEEPose) action publishing.

Deliberately NOT part of ``ActionLimiter``: that class is built entirely
around named joint positions and upstream's ``ensure_safe_goal_position``
(per-joint max_relative_target against the *previous commanded* position).
None of that applies to an [x, y, z, qx, qy, qz, qw, gripper] vector — there
is no "previous joint position" to diff against, and clamping a quaternion
component independently produces an invalid (non-unit) rotation, not a
"safer" one.

What this class actually does, and does NOT do:
  - Renormalizes the quaternion to unit length. This is NOT a safety
    feature — a non-unit quaternion is not a valid rotation at all, and raw
    model output has no norm=1 guarantee. Skipping this would send malformed
    commands regardless of magnitude.
  - Clamps the gripper to CommandedEEPose.msg's documented physical range
    ([-0.003, 0.05] m).
  - Clamps per-tick POSITION delta magnitude to `max_position_delta_m`
    (metres), relative to the last position this limiter published (not
    read from the robot — this class has no observation channel back in).

What it explicitly does NOT do:
  - No per-tick ROTATION delta limit. I have no validated safe bound for
    this robot's Cartesian rotation rate at the inference control frequency,
    and fabricating one would be worse than an honest gap: an unreviewed
    "safety" limit that happens to be wrong is more dangerous than a known
    absence, because it invites trusting the wrong thing.
  - No comparison against the robot's OWN current EE pose (e.g. from
    /ee_pose_{arm} readback) — only against this limiter's own last output.
    A large first-command jump from wherever the arm physically is right now
    is NOT caught by this class.

`max_position_delta_m` defaults conservatively (5 mm/tick) but this default
is UNVALIDATED against real hardware — reviewed and tuned by whoever runs
this for the first time, not assumed safe because it's the default.
"""

from __future__ import annotations

import numpy as np


class EEPoseLimiter:
    """Per-arm rate limiter for absolute EE pose + gripper actions.

    Args:
        max_position_delta_m: Maximum allowed change in position (xyz, metres)
            between consecutive published commands for this arm. None disables
            position limiting entirely (echo/testing only — do not disable for
            a real run without a specific reason).
        gripper_min: Physical gripper lower bound (metres), from
            CommandedEEPose.msg's documented range.
        gripper_max: Physical gripper upper bound (metres).
    """

    def __init__(
        self,
        max_position_delta_m: float | None = 0.005,
        gripper_min: float = -0.003,
        gripper_max: float = 0.05,
    ) -> None:
        self.max_position_delta_m = max_position_delta_m
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self._last_position: np.ndarray | None = None
        self._clamp_count = 0
        self._tick_count = 0

    def process(self, ee_action: np.ndarray) -> np.ndarray:
        """Limit one [x, y, z, qx, qy, qz, qw, gripper] action.

        Returns a new array; does not modify ee_action in place.
        """
        ee_action = np.asarray(ee_action, dtype=np.float64).copy()
        if ee_action.shape[-1] != 8:
            raise ValueError(
                f"EEPoseLimiter expects an 8-element [x,y,z,qx,qy,qz,qw,gripper] "
                f"action, got shape {ee_action.shape}"
            )
        self._tick_count += 1

        position = ee_action[0:3]
        quat = ee_action[3:7]
        gripper = ee_action[7]

        # Renormalize the quaternion -- correctness, not safety. A near-zero
        # norm (degenerate model output) falls back to identity rather than
        # dividing by ~0.
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            quat = quat / norm
        else:
            quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Position rate limit, relative to this limiter's own last output —
        # NOT the robot's actual current pose (no readback wired in here).
        if self.max_position_delta_m is not None and self._last_position is not None:
            delta = position - self._last_position
            delta_norm = np.linalg.norm(delta)
            if delta_norm > self.max_position_delta_m:
                position = self._last_position + delta * (self.max_position_delta_m / delta_norm)
                self._clamp_count += 1

        self._last_position = position.copy()

        gripper = float(np.clip(gripper, self.gripper_min, self.gripper_max))

        return np.concatenate([position, quat, [gripper]])

    def reset(self) -> None:
        """Clear rate-limit history — call when (re)starting a run so the
        first command after (re)start isn't rate-limited against a stale pose."""
        self._last_position = None

    def get_clamp_rate(self) -> float:
        """Fraction of ticks where the position delta was clamped."""
        return self._clamp_count / self._tick_count if self._tick_count else 0.0
