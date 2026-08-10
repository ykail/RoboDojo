"""Small, mockable dual-ARX-X5 hardware backend.

The vendor ``arx5_interface`` module is intentionally imported only when
``connect`` is called.  Unit tests can inject a factory returning a fake module,
while the real adapter uses exactly the API already exercised by Robot_Lab.
"""

from __future__ import annotations

import importlib
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable


SIDES = ("left", "right")
JOINT_DOF = 6


class X5HardwareError(RuntimeError):
    """Raised when the dual-X5 backend cannot complete a hardware operation."""


class X5Mode(str, Enum):
    DISCONNECTED = "disconnected"
    HOLD = "hold"
    FOLLOW = "follow"
    TEACH = "teach"
    FAULT = "fault"


@dataclass(frozen=True)
class ArmState:
    q_rad: tuple[float, ...]
    gripper_pos_m: float
    gripper_open_fraction: float
    sample_monotonic_ns: int
    seq: int


@dataclass(frozen=True)
class DualState:
    left: ArmState
    right: ArmState

    def side(self, name: str) -> ArmState:
        if name not in SIDES:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True)
class ArmTarget:
    q_rad: tuple[float, ...]
    gripper_open_fraction: float


@dataclass(frozen=True)
class DualTarget:
    left: ArmTarget
    right: ArmTarget

    def side(self, name: str) -> ArmTarget:
        if name not in SIDES:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True)
class X5HardwareConfig:
    left_model: str = "X5"
    right_model: str = "X5"
    left_can: str = "can1"
    right_can: str = "can3"
    clear_on_init: bool = False
    gravity_compensation: bool = True

    active_joint_kp: tuple[float, ...] = (90.0, 80.0, 80.0, 35.0, 35.0, 25.0)
    active_joint_kd: tuple[float, ...] = (2.5, 2.2, 2.2, 1.6, 1.6, 1.0)
    teach_joint_kd: tuple[float, ...] = (0.3, 0.3, 0.3, 0.2, 0.1, 0.1)

    # Isaac supplies one joint target every 40 ms.  A zero-timestamp ARX
    # command replaces the SDK interpolator with an instantaneous setpoint,
    # which makes a nominally smooth 25 Hz trajectory visibly step.  Schedule
    # each follow waypoint one control frame ahead so the SDK's 500 Hz thread
    # linearly interpolates between consecutive targets.
    follow_preview_s: float = 0.04

    left_gripper_kp: float = 2.0
    left_gripper_kd: float = 0.15
    right_gripper_kp: float = 3.0
    right_gripper_kd: float = 0.2
    teach_gripper_kp: float = 0.0
    teach_gripper_kd: float = 0.0

    left_gripper_min_m: float = 0.0
    left_gripper_max_m: float | None = None
    right_gripper_min_m: float = 0.0
    right_gripper_max_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("active_joint_kp", "active_joint_kd", "teach_joint_kd"):
            values = tuple(float(value) for value in getattr(self, name))
            if len(values) != JOINT_DOF or not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(f"{name} must contain six finite non-negative values")
        for name in (
            "left_gripper_kp",
            "left_gripper_kd",
            "right_gripper_kp",
            "right_gripper_kd",
            "teach_gripper_kp",
            "teach_gripper_kd",
            "left_gripper_min_m",
            "right_gripper_min_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("left_gripper_max_m", "right_gripper_max_m"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} must be positive when supplied")
        if not math.isfinite(float(self.follow_preview_s)) or self.follow_preview_s < 0:
            raise ValueError("follow_preview_s must be finite and non-negative")


def _load_arx5_interface() -> Any:
    try:
        return importlib.import_module("arx5_interface")
    except ModuleNotFoundError as exc:
        raise X5HardwareError(
            "arx5_interface is unavailable; run this source in the verified ARX X5 SDK environment"
        ) from exc


class DualX5Hardware:
    """Own two X5 controllers and serialize all SDK access in the caller thread."""

    def __init__(
        self,
        config: X5HardwareConfig,
        *,
        sdk_factory: Callable[[], Any] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sdk_factory = sdk_factory or _load_arx5_interface
        self._monotonic_ns = monotonic_ns
        self._sleep = sleep
        self._sdk: Any | None = None
        self._controllers: dict[str, Any] = {}
        self._robot_configs: dict[str, Any] = {}
        self._controller_configs: dict[str, Any] = {}
        self._gripper_ranges: dict[str, tuple[float, float]] = {}
        self._mode = X5Mode.DISCONNECTED
        self._latched_target: DualTarget | None = None
        self._sample_seq = 0

    @property
    def mode(self) -> X5Mode:
        return self._mode

    @property
    def is_connected(self) -> bool:
        return self._sdk is not None and set(self._controllers) == set(SIDES)

    @property
    def latched_target(self) -> DualTarget:
        if self._latched_target is None:
            raise X5HardwareError("dual X5 has no latched target")
        return self._latched_target

    def gripper_range(self, side: str) -> tuple[float, float]:
        if side not in self._gripper_ranges:
            raise X5HardwareError(f"{side} X5 gripper range is unavailable")
        return self._gripper_ranges[side]

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise X5HardwareError("dual X5 hardware is not connected")
        if self._mode == X5Mode.FAULT:
            raise X5HardwareError("dual X5 hardware is faulted")

    def _controller_parameters(self, side: str) -> tuple[str, str]:
        if side == "left":
            return self.config.left_model, self.config.left_can
        if side == "right":
            return self.config.right_model, self.config.right_can
        raise KeyError(side)

    def _resolve_gripper_range(self, side: str, robot_config: Any) -> tuple[float, float]:
        low = float(getattr(self.config, f"{side}_gripper_min_m"))
        configured_high = getattr(self.config, f"{side}_gripper_max_m")
        high = float(robot_config.gripper_width if configured_high is None else configured_high)
        if not math.isfinite(low) or not math.isfinite(high) or low < 0 or high <= low:
            raise X5HardwareError(f"invalid {side} X5 gripper range [{low}, {high}] m")
        return low, high

    def connect(self) -> DualState:
        if self.is_connected:
            raise X5HardwareError("dual X5 hardware is already connected")
        self._sdk = self._sdk_factory()
        created: list[Any] = []
        try:
            robot_factory = self._sdk.RobotConfigFactory.get_instance()
            controller_factory = self._sdk.ControllerConfigFactory.get_instance()
            for side in SIDES:
                model, interface = self._controller_parameters(side)
                robot_config = robot_factory.get_config(model)
                if int(robot_config.joint_dof) != JOINT_DOF:
                    raise X5HardwareError(
                        f"{side} model {model!r} has {robot_config.joint_dof} joints; expected six"
                    )
                controller_config = controller_factory.get_config("joint_controller", JOINT_DOF)
                controller_config.background_send_recv = True
                if hasattr(controller_config, "clear_motor_state_on_init"):
                    controller_config.clear_motor_state_on_init = self.config.clear_on_init
                if hasattr(controller_config, "gravity_compensation"):
                    controller_config.gravity_compensation = self.config.gravity_compensation
                controller = self._sdk.Arx5JointController(
                    robot_config,
                    controller_config,
                    interface,
                )
                created.append(controller)
                if hasattr(controller, "set_log_level") and hasattr(self._sdk, "LogLevel"):
                    controller.set_log_level(self._sdk.LogLevel.WARNING)
                self._robot_configs[side] = robot_config
                self._controller_configs[side] = controller_config
                self._controllers[side] = controller
                self._gripper_ranges[side] = self._resolve_gripper_range(side, robot_config)
            self._mode = X5Mode.HOLD
            return self.enter_hold()
        except BaseException as exc:
            for controller in created:
                try:
                    controller.set_to_damping()
                except Exception:
                    pass
            self._controllers.clear()
            self._robot_configs.clear()
            self._controller_configs.clear()
            self._gripper_ranges.clear()
            self._sdk = None
            self._mode = X5Mode.DISCONNECTED
            if isinstance(exc, X5HardwareError):
                raise
            raise X5HardwareError(f"failed to connect dual X5 hardware: {exc}") from exc

    def _read_side(self, side: str, sample_ns: int, seq: int) -> ArmState:
        raw = self._controllers[side].get_joint_state()
        q = tuple(float(value) for value in raw.pos())
        gripper_pos = float(raw.gripper_pos)
        low, high = self._gripper_ranges[side]
        if len(q) != JOINT_DOF or not all(math.isfinite(value) for value in q):
            raise X5HardwareError(f"invalid {side} X5 joint feedback")
        if not math.isfinite(gripper_pos):
            raise X5HardwareError(f"invalid {side} X5 gripper feedback")
        fraction = min(1.0, max(0.0, (gripper_pos - low) / (high - low)))
        return ArmState(q, gripper_pos, fraction, sample_ns, seq)

    def read(self) -> DualState:
        self._require_connected()
        try:
            sample_ns = int(self._monotonic_ns())
            self._sample_seq += 1
            left = self._read_side("left", sample_ns, self._sample_seq)
            right = self._read_side("right", sample_ns, self._sample_seq)
            return DualState(left, right)
        except BaseException as exc:
            if isinstance(exc, X5HardwareError):
                raise
            raise X5HardwareError(f"failed to read dual X5 state: {exc}") from exc

    @staticmethod
    def target_from_state(state: DualState) -> DualTarget:
        return DualTarget(
            ArmTarget(state.left.q_rad, state.left.gripper_open_fraction),
            ArmTarget(state.right.q_rad, state.right.gripper_open_fraction),
        )

    def _normalize_target(self, target: DualTarget) -> DualTarget:
        normalized: dict[str, ArmTarget] = {}
        for side in SIDES:
            item = target.side(side)
            q = tuple(float(value) for value in item.q_rad)
            fraction = float(item.gripper_open_fraction)
            if len(q) != JOINT_DOF or not all(math.isfinite(value) for value in q):
                raise X5HardwareError(f"invalid {side} X5 joint target")
            if not math.isfinite(fraction):
                raise X5HardwareError(f"invalid {side} X5 gripper target")
            robot_config = self._robot_configs[side]
            lower = tuple(float(value) for value in robot_config.joint_pos_min[:JOINT_DOF])
            upper = tuple(float(value) for value in robot_config.joint_pos_max[:JOINT_DOF])
            clipped = tuple(min(high, max(low, value)) for value, low, high in zip(q, lower, upper))
            normalized[side] = ArmTarget(clipped, min(1.0, max(0.0, fraction)))
        return DualTarget(normalized["left"], normalized["right"])

    def _sdk_joint_state(
        self,
        side: str,
        target: ArmTarget,
        *,
        preview_s: float = 0.0,
    ) -> Any:
        assert self._sdk is not None
        command = self._sdk.JointState(JOINT_DOF)
        command.pos()[:] = target.q_rad
        low, high = self._gripper_ranges[side]
        command.gripper_pos = low + target.gripper_open_fraction * (high - low)
        if preview_s > 0.0:
            controller_time = float(self._controllers[side].get_timestamp())
            if not math.isfinite(controller_time):
                raise X5HardwareError(f"invalid {side} X5 controller timestamp")
            command.timestamp = controller_time + preview_s
        return command

    def _write(self, target: DualTarget, *, preview_s: float = 0.0) -> DualTarget:
        if not math.isfinite(float(preview_s)) or preview_s < 0.0:
            raise X5HardwareError("X5 command preview must be finite and non-negative")
        applied = self._normalize_target(target)
        try:
            for side in SIDES:
                self._controllers[side].set_joint_cmd(
                    self._sdk_joint_state(
                        side,
                        applied.side(side),
                        preview_s=preview_s,
                    )
                )
        except BaseException as exc:
            self._mode = X5Mode.FAULT
            for controller in self._controllers.values():
                try:
                    controller.set_to_damping()
                except Exception:
                    pass
            raise X5HardwareError(f"failed to command dual X5 hardware: {exc}") from exc
        self._latched_target = applied
        return applied

    def _set_gains(self, teach: bool) -> None:
        assert self._sdk is not None
        for side in SIDES:
            gain = self._sdk.Gain(JOINT_DOF)
            if teach:
                gain.kp()[:] = (0.0,) * JOINT_DOF
                gain.kd()[:] = self.config.teach_joint_kd
                gain.gripper_kp = self.config.teach_gripper_kp
                gain.gripper_kd = self.config.teach_gripper_kd
            else:
                gain.kp()[:] = self.config.active_joint_kp
                gain.kd()[:] = self.config.active_joint_kd
                gain.gripper_kp = float(getattr(self.config, f"{side}_gripper_kp"))
                gain.gripper_kd = float(getattr(self.config, f"{side}_gripper_kd"))
            self._controllers[side].set_gain(gain)

    def enter_hold(self) -> DualState:
        self._require_connected()
        current = self.read()
        current_target = self.target_from_state(current)
        # Overwrite any old SDK target before restoring position gains, then
        # write the same target once more with active gains.  This ordering is
        # what prevents a teach -> follow transition from jumping backwards.
        self._write(current_target)
        self._set_gains(teach=False)
        self._write(current_target)
        self._mode = X5Mode.HOLD
        return current

    def follow(self, target: DualTarget) -> DualTarget:
        self._require_connected()
        if self._mode not in {X5Mode.HOLD, X5Mode.FOLLOW}:
            raise X5HardwareError(f"cannot follow while dual X5 is in {self._mode.value} mode")
        applied = self._write(target, preview_s=self.config.follow_preview_s)
        self._mode = X5Mode.FOLLOW
        return applied

    def hold(self) -> DualTarget:
        self._require_connected()
        if self._mode == X5Mode.TEACH:
            self.enter_hold()
        target = self.latched_target
        applied = self._write(target)
        self._mode = X5Mode.HOLD
        return applied

    def enter_teach(self) -> DualState:
        self._require_connected()
        current = self.read()
        current_target = self.target_from_state(current)
        self._write(current_target)
        try:
            for side in SIDES:
                self._controllers[side].set_to_damping()
            self._set_gains(teach=True)
            # Keeps the gripper command coherent when teach_gripper_kp is
            # non-zero; joint kp is zero, so this cannot pull an arm backwards.
            self._write(current_target)
        except BaseException as exc:
            self._mode = X5Mode.FAULT
            raise X5HardwareError(f"failed to enter dual X5 teach mode: {exc}") from exc
        self._mode = X5Mode.TEACH
        return self.read()

    def move_home(
        self,
        q_rad: tuple[float, ...],
        gripper_open_fraction: float,
        *,
        duration_s: float,
        frequency_hz: float,
    ) -> DualTarget:
        self._require_connected()
        if len(q_rad) != JOINT_DOF or not all(math.isfinite(float(value)) for value in q_rad):
            raise X5HardwareError("home q_rad must contain six finite values")
        if not math.isfinite(gripper_open_fraction):
            raise X5HardwareError("home gripper fraction must be finite")
        if duration_s < 0 or frequency_hz <= 0:
            raise X5HardwareError("home duration must be non-negative and frequency positive")
        start = self.enter_hold()
        goal_arm = ArmTarget(tuple(float(value) for value in q_rad), gripper_open_fraction)
        goal = DualTarget(goal_arm, goal_arm)
        steps = max(1, int(round(duration_s * frequency_hz)))
        for step in range(1, steps + 1):
            linear = step / steps
            alpha = linear * linear * (3.0 - 2.0 * linear)
            arms: dict[str, ArmTarget] = {}
            for side in SIDES:
                initial = start.side(side)
                arms[side] = ArmTarget(
                    tuple(a + alpha * (b - a) for a, b in zip(initial.q_rad, goal_arm.q_rad)),
                    initial.gripper_open_fraction
                    + alpha * (goal_arm.gripper_open_fraction - initial.gripper_open_fraction),
                )
            self._write(
                DualTarget(arms["left"], arms["right"]),
                preview_s=(duration_s / steps if duration_s > 0 else 0.0),
            )
            self._mode = X5Mode.FOLLOW
            if duration_s > 0:
                self._sleep(duration_s / steps)
        applied = self._write(goal)
        self._mode = X5Mode.HOLD
        return applied

    def close(self) -> None:
        if not self.is_connected:
            self._mode = X5Mode.DISCONNECTED
            return
        try:
            if self._mode != X5Mode.FAULT:
                self.enter_hold()
        except Exception:
            pass
        for controller in self._controllers.values():
            try:
                controller.set_to_damping()
            except Exception:
                pass
        self._controllers.clear()
        self._robot_configs.clear()
        self._controller_configs.clear()
        self._gripper_ranges.clear()
        self._latched_target = None
        self._sdk = None
        self._mode = X5Mode.DISCONNECTED
