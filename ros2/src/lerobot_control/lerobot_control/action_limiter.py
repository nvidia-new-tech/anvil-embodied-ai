"""Action limiter for safe robot control.

Follows upstream LeRobot's ``openarm_follower.send_action``
(lerobot/robots/openarm_follower/openarm_follower.py:274-330) as closely as this
transport allows:

    goal_pos = {name: value, ...}                    # named, order-independent
    goal_pos = <clip to joint_limits>                # layer 1
    goal_pos = ensure_safe_goal_position(...)        # layer 2 (imported upstream)

Upstream works in ``RobotAction`` dicts keyed by motor name (``"joint_1.pos"``),
so joint ORDER never enters the picture — names carry the mapping. This node
publishes ``Float64MultiArray``, a bare positional array, so the array/name
conversion has to happen somewhere. Doing it at the edges of this class means
everything in between is the upstream code path, and reordering falls out of
``to_array`` rather than being separate index bookkeeping.

The per-joint form of ``max_relative_target`` matters here because the action
vector MIXES UNITS: joint1..joint7 are radians while finger_joint1 is prismatic
in metres over a ~0.05 m range. One scalar cannot serve both — 0.1 throttles the
arm to 7-18 steps of travel while exceeding the gripper's entire range, so the
gripper shuts in a single step while the arm is still descending.
"""

import logging
import time

import numpy as np

try:
    from lerobot.robots.utils import ensure_safe_goal_position
except ImportError as e:  # pragma: no cover - lerobot is a hard dependency
    raise ImportError(
        "ActionLimiter uses lerobot.robots.utils.ensure_safe_goal_position so that "
        "delta capping matches upstream exactly. Install lerobot."
    ) from e


class _SuppressClampSpam(logging.Filter):
    """Drops upstream's per-clamp warning and counts it instead.

    ``ensure_safe_goal_position`` calls ``logging.warning`` with a pformat dump
    every time it clamps anything. That is reasonable at teleop rates, but this
    node runs inference at 30 Hz and the arm legitimately clamps on most steps
    during a fast approach — measured at 85%. Left alone it emits a multi-line
    dump 25+ times a second and buries everything else.

    We drop the record and let ActionLimiter emit a rate-limited summary that
    carries the same information.
    """

    NEEDLE = "Relative goal position magnitude had to be clamped"

    def __init__(self):
        super().__init__()
        self.count = 0

    def filter(self, record: logging.LogRecord) -> bool:
        if self.NEEDLE in str(record.msg):
            self.count += 1
            return False
        return True


class ActionLimiter:
    """Applies upstream safety limits, converting to/from positional arrays.

    Args:
        max_relative_target: Per-step displacement cap. A float caps every joint
            equally; a dict maps joint name -> cap. Upstream requires a dict to
            name EVERY joint (``ensure_safe_goal_position`` raises otherwise),
            which is deliberate: a silently uncapped joint is worse than a loud
            config error. None disables the layer.
        joint_limits: Joint name -> (min, max) absolute bounds. None disables the
            layer. No defaults are assumed — upstream's are for its own bus and
            units and would be wrong here, where joints are radians and the
            gripper is metres.
        model_joint_order: Order the ML model outputs actions.
        controller_joint_order: Order the ROS2 controller expects.
        delta_exclude_joints: Joint names kept in absolute space (not delta-restored).
        logger: Optional ROS2 logger.
    """

    def __init__(
        self,
        max_relative_target: float | dict[str, float] | None = None,
        joint_limits: dict[str, tuple[float, float]] | None = None,
        model_joint_order: list[str] | None = None,
        controller_joint_order: list[str] | None = None,
        delta_exclude_joints: list[str] | None = None,
        logger=None,
    ):
        self.model_joint_order = model_joint_order or []
        self.controller_joint_order = controller_joint_order or self.model_joint_order
        self.logger = logger

        # Upstream type-checks `isinstance(x, float)`, so an int from YAML would
        # fall through to its TypeError. Normalise here instead.
        if isinstance(max_relative_target, (int, float)) and not isinstance(
            max_relative_target, bool
        ):
            max_relative_target = float(max_relative_target)
        elif isinstance(max_relative_target, dict):
            max_relative_target = {k: float(v) for k, v in max_relative_target.items()}
        elif max_relative_target is not None:
            raise TypeError(
                f"max_relative_target must be a float, a dict, or None; "
                f"got {type(max_relative_target).__name__}"
            )
        self.max_relative_target = max_relative_target

        self.joint_limits = (
            {k: (float(v[0]), float(v[1])) for k, v in joint_limits.items()}
            if joint_limits
            else None
        )

        self._validate()

        ref = self.controller_joint_order
        self._delta_exclude_indices: set[int] = {
            ref.index(name) for name in (delta_exclude_joints or []) if name in ref
        }

        # Upstream logs on every clamp; see _SuppressClampSpam.
        self._clamp_filter = _SuppressClampSpam()
        logging.getLogger().addFilter(self._clamp_filter)
        self._clamp_report_interval = 5.0
        self._clamp_last_report = time.monotonic()
        self._clamp_steps = 0
        self._steps = 0

    def _validate(self) -> None:
        """Fail at construction rather than mid-episode."""
        model, ctrl = self.model_joint_order, self.controller_joint_order
        if model and ctrl and set(model) != set(ctrl):
            raise ValueError(
                f"model_joint_order and controller_joint_order must contain the same "
                f"names; model-only={sorted(set(model) - set(ctrl))}, "
                f"controller-only={sorted(set(ctrl) - set(model))}"
            )

        if isinstance(self.max_relative_target, dict) and ctrl:
            # Mirror ensure_safe_goal_position's exact-match rule, but report it
            # now with a usable message instead of on the first published action.
            if set(self.max_relative_target) != set(ctrl):
                missing = sorted(set(ctrl) - set(self.max_relative_target))
                extra = sorted(set(self.max_relative_target) - set(ctrl))
                raise ValueError(
                    f"max_relative_target must name every joint (upstream "
                    f"ensure_safe_goal_position requires an exact key match). "
                    f"missing={missing}, unknown={extra}"
                )

        if self.joint_limits and ctrl:
            unknown = sorted(set(self.joint_limits) - set(ctrl))
            if unknown:
                raise ValueError(f"joint_limits names not in joint order: {unknown}")
            for name, (lo, hi) in self.joint_limits.items():
                if lo > hi:
                    raise ValueError(f"joint_limits['{name}']: min {lo} > max {hi}")

    def reset(self) -> None:
        """Reset per-episode state. Both layers are stateless; kept for the call
        contract used on episode boundaries and model reload."""
        return

    def _log(self, level: str, msg: str):
        if self.logger:
            getattr(self.logger, level)(msg)
        else:
            print(f"[{level.upper()}] {msg}")

    # --- array <-> name conversion (the only part upstream does not need) ----

    def to_named(self, action: np.ndarray, order: list[str] | None = None) -> dict[str, float]:
        """Positional array -> ``{joint_name: value}``, upstream's representation."""
        order = order or self.model_joint_order
        if not order or len(action) != len(order):
            raise ValueError(
                f"cannot name a {len(action)}-element action with a "
                f"{len(order)}-name joint order"
            )
        return {name: float(v) for name, v in zip(order, action)}

    def to_array(self, goal_pos: dict[str, float]) -> np.ndarray:
        """``{joint_name: value}`` -> array in controller order.

        Reordering is implicit: values are placed by NAME, so a model order of
        [finger, j1..j7] and a controller order of [j1..j7, finger] need no
        separate index mapping.
        """
        order = self.controller_joint_order
        missing = set(order) - set(goal_pos)
        if missing:
            raise ValueError(f"missing joints for controller order: {sorted(missing)}")
        return np.array([goal_pos[name] for name in order], dtype=float)

    def reorder(self, action: np.ndarray) -> np.ndarray:
        """Model order -> controller order. Kept for callers that want only this."""
        if not self.model_joint_order or not self.controller_joint_order:
            return action
        if self.model_joint_order == self.controller_joint_order:
            return action
        if len(action) != len(self.model_joint_order):
            return action
        return self.to_array(self.to_named(action))

    # --- the two upstream layers --------------------------------------------

    def apply_joint_limits(self, goal_pos: dict[str, float]) -> dict[str, float]:
        """Clip to absolute bounds. Verbatim from openarm_follower.py:277-283."""
        if not self.joint_limits:
            return goal_pos
        goal_pos = dict(goal_pos)
        for motor_name, position in goal_pos.items():
            if motor_name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[motor_name]
                clipped_position = max(min_limit, min(max_limit, position))
                if clipped_position != position:
                    self._log(
                        "debug",
                        f"Clipped {motor_name} from {position:.4f} to {clipped_position:.4f}",
                    )
                goal_pos[motor_name] = clipped_position
        return goal_pos

    def apply_delta_limit(
        self, goal_pos: dict[str, float], present_pos: dict[str, float]
    ) -> dict[str, float]:
        """Cap per-step displacement. Delegates to upstream, as openarm_follower
        does at :287-290, so the capping maths is never a local reimplementation."""
        if self.max_relative_target is None:
            return goal_pos
        goal_present_pos = {k: (g, present_pos[k]) for k, g in goal_pos.items() if k in present_pos}
        if len(goal_present_pos) != len(goal_pos):
            return goal_pos

        before = self._clamp_filter.count
        capped = ensure_safe_goal_position(goal_present_pos, self.max_relative_target)

        self._steps += 1
        if self._clamp_filter.count > before:
            self._clamp_steps += 1
        self._maybe_report_clamping()
        return capped

    def _maybe_report_clamping(self) -> None:
        """Rate-limited stand-in for the per-call warning we filtered out."""
        now = time.monotonic()
        if now - self._clamp_last_report < self._clamp_report_interval:
            return
        if self._clamp_steps:
            pct = 100.0 * self._clamp_steps / max(self._steps, 1)
            self._log(
                "warn",
                f"max_relative_target clamped {self._clamp_steps}/{self._steps} steps "
                f"({pct:.0f}%) in the last {self._clamp_report_interval:.0f}s — the arm is "
                f"rate-limited and lags the model's target",
            )
        self._clamp_last_report = now
        self._clamp_steps = 0
        self._steps = 0

    def process(
        self,
        action: np.ndarray,
        current_positions: np.ndarray | None = None,
        joint_order: list[str] | None = None,
        ref_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Name the action, apply both upstream layers, return controller order.

        Args:
            action: Absolute action in MODEL joint order, already delta-restored
                upstream by inference_node.
            current_positions: Current positions in CONTROLLER joint order.
            joint_order: Unused, kept for backward compatibility.
            ref_state: Unused, kept for backward compatibility.

        Returns:
            Action in controller order, ready to publish.
        """
        goal_pos = self.to_named(action)

        # Layer 1 — absolute per-joint bounds
        goal_pos = self.apply_joint_limits(goal_pos)

        # Layer 2 — per-step displacement cap, against the live position
        if current_positions is not None and len(current_positions) == len(
            self.controller_joint_order
        ):
            present = self.to_named(current_positions, self.controller_joint_order)
            goal_pos = self.apply_delta_limit(goal_pos, present)

        return self.to_array(goal_pos)

    def get_clamped_joints(
        self, action: np.ndarray, current_positions: np.ndarray
    ) -> list[int]:
        """Indices (controller order) whose displacement would be capped."""
        if self.max_relative_target is None or current_positions is None:
            return []
        try:
            goal = self.to_named(action)
            present = self.to_named(current_positions, self.controller_joint_order)
        except ValueError:
            return []
        capped = self.apply_delta_limit(goal, present)
        return [
            i
            for i, name in enumerate(self.controller_joint_order)
            if abs(capped[name] - goal[name]) > 1e-9
        ]
