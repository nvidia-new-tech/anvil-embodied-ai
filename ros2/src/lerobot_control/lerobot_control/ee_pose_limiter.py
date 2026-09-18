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
        position_offset_m: Constant [dx, dy, dz] in metres added to every
            commanded position, in the same `world` frame the pose is
            expressed in. None or all-zero disables it.

            This is a deployment-side correction, not a policy fix. Use it
            when the arm reproduces the policy's trajectory faithfully but
            consistently lands shallow or off to one side -- an offset
            between the frame the demonstrations were recorded in and where
            this rig actually is. It cannot fix a policy that reaches the
            wrong place for the wrong reason, and it moves EVERY command
            including the retreats, so a downward dz presses the whole
            trajectory toward the table, not only the grasp.

            No limit is enforced on the magnitude here: a value large enough
            to drive the arm into the table is a value this class will
            happily apply. Verify with a dry run (publish to an unsubscribed
            topic and read the commands back) before trusting one.
        position_scale: Per-axis gain [sx, sy, sz] on displacement away from
            the first commanded position of the run. None or all-ones
            disables it.

                out = pivot + (commanded - pivot) * scale

            The pivot is the first position this limiter sees after
            construction or reset(), so the first command is unchanged and
            later excursions grow around it. Scaling the raw coordinate
            instead would just translate the trajectory -- 0.50 m * 1.2 is
            0.60 m, an arm 10 cm higher, not a deeper reach.

            A gain above 1 is a much larger departure from the policy than
            position_offset_m: it stretches the trajectory in both
            directions, so a reach goes deeper AND the retreat goes higher,
            and it amplifies whatever tracking error and noise the policy
            already has. The commanded poses leave the training range the
            policy was validated over, so the guarantee that they were
            reachable and collision-free goes with it.

            It also multiplies consecutive-command deltas by the same gain,
            so max_position_delta_m clamps proportionally more often.

            Reach for position_offset_m first: if the arm is uniformly
            shallow, that is a frame offset and an offset fixes it without
            distorting anything. A gain only makes sense when the SHAPE of
            the motion is too small -- the arm dips, just not far enough.
    """

    def __init__(
        self,
        max_position_delta_m: float | None = 0.005,
        gripper_min: float = -0.003,
        gripper_max: float = 0.05,
        position_offset_m: tuple[float, float, float] | list[float] | None = None,
        position_scale: tuple[float, float, float] | list[float] | None = None,
    ) -> None:
        self.max_position_delta_m = max_position_delta_m
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        if position_offset_m is None:
            self.position_offset_m = None
        else:
            offset = np.asarray(position_offset_m, dtype=np.float64).ravel()
            if offset.shape != (3,):
                raise ValueError(
                    f"position_offset_m must be 3 elements [dx, dy, dz], "
                    f"got shape {offset.shape}"
                )
            # Treat an explicit all-zero offset as "off" so the arithmetic and
            # the log line below are skipped rather than adding 0.0 every tick.
            self.position_offset_m = offset if np.any(offset) else None

        if position_scale is None:
            self.position_scale = None
        else:
            scale = np.asarray(position_scale, dtype=np.float64).ravel()
            if scale.shape != (3,):
                raise ValueError(
                    f"position_scale must be 3 elements [sx, sy, sz], "
                    f"got shape {scale.shape}"
                )
            # All-ones is identity; skip the arithmetic and the pivot bookkeeping.
            self.position_scale = scale if not np.allclose(scale, 1.0) else None
        self._scale_pivot: np.ndarray | None = None
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

        # Gain on displacement from where this run started. The pivot is
        # captured from the first command rather than configured, so the
        # stretch is anchored to wherever the arm actually began instead of a
        # number that goes stale when the home pose changes.
        if self.position_scale is not None:
            if self._scale_pivot is None:
                self._scale_pivot = position.copy()
            position = self._scale_pivot + (position - self._scale_pivot) * self.position_scale

        # Constant frame correction, applied before the rate limit so the
        # clamp sees the position actually being commanded. Being constant it
        # cancels out of consecutive-command deltas, so it does not change how
        # often the clamp fires -- except on the very first command, which is
        # not clamped at all (_last_position is None), and therefore takes the
        # whole offset in one step.
        if self.position_offset_m is not None:
            position = position + self.position_offset_m

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
        # Drop the scale pivot too: it anchors to the start of a run, and a
        # stale one from the previous run would stretch the new trajectory
        # around a pose the arm is no longer at.
        self._scale_pivot = None

    def get_clamp_rate(self) -> float:
        """Fraction of ticks where the position delta was clamped."""
        return self._clamp_count / self._tick_count if self._tick_count else 0.0
