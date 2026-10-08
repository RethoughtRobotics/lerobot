"""
Simulated Piper robot for testing joint mapping without hardware.
Applies SDK joint limits and logs commanded vs clamped positions.

Usage with replay:
    uv run python -m lerobot.scripts.lerobot_replay \
        --robot.type=piper_sim \
        --dataset.repo_id=CaptainRapid/TestTubeInsertion \
        --dataset.root=$HOME/.cache/huggingface/lerobot/CaptainRapid/TestTubeInsertion \
        --dataset.episode=0
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation

from ..config import RobotConfig
from ..robot import Robot

logger = logging.getLogger(__name__)

# SDK joint limits in degrees (from Agilex API docs)
SDK_LIMITS = {
    "joint1": (-150.0, 150.0),
    "joint2": (0.0,    180.0),
    "joint3": (-170.0,   0.0),
    "joint4": (-100.0, 100.0),
    "joint5": (-70.0,   70.0),
    "joint6": (-120.0, 120.0),
    "gripper": (0.0,   100.0),
}

MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


@RobotConfig.register_subclass("piper_sim")
@dataclass
class PiperSimConfig(RobotConfig):
    swap_joints_2_3: bool = True  # toggle to compare with/without swap


class PiperRobotSim(Robot):
    """Simulated bimanual Piper — no hardware required."""

    config_class = PiperSimConfig
    name = "piper_sim"

    def __init__(self, config: PiperSimConfig):
        super().__init__(config)
        self.config = config
        self._state = {f"left_{m}.pos": 0.0 for m in MOTOR_NAMES}
        self._state.update({f"right_{m}.pos": 0.0 for m in MOTOR_NAMES})
        self._is_connected = False
        self._clamp_counts = {k: 0 for k in self._state}
        self._frame = 0

    @cached_property
    def observation_features(self) -> dict:
        return {k: float for k in self._state}

    @cached_property
    def action_features(self) -> dict:
        return {k: float for k in self._state}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._is_connected = True
        logger.info(f"[SIM] Connected (swap_joints_2_3={self.config.swap_joints_2_3})")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        return dict(self._state)

    def _sdk_joints(self, arm_vals: list[float]) -> list[float]:
        """Apply joint2/3 swap if enabled, matching write() in piper.py."""
        if self.config.swap_joints_2_3:
            # dataset: [j1, j2, j3, j4, j5, j6, gripper]
            # sdk:     [j1, j3, j2, j4, j5, j6, gripper]
            return [arm_vals[0], arm_vals[2], arm_vals[1],
                    arm_vals[3], arm_vals[4], arm_vals[5], arm_vals[6]]
        return list(arm_vals)

    def _apply_limits(self, sdk_vals: list[float], side: str) -> tuple[list[float], list[str]]:
        clamped_joints = []
        result = []
        for i, (name, val) in enumerate(zip(MOTOR_NAMES, sdk_vals)):
            lo, hi = SDK_LIMITS[name]
            clamped = float(np.clip(val, lo, hi))
            if abs(clamped - val) > 0.01:
                clamped_joints.append(
                    f"{side}_{name}: cmd={val:.2f} clamped to {clamped:.2f} (limit [{lo},{hi}])"
                )
            result.append(clamped)
        return result, clamped_joints

    def send_action(self, action: RobotAction) -> RobotAction:
        for side, prefix in [("left", "left"), ("right", "right")]:
            dataset_vals = [float(action[f"{prefix}_{m}.pos"]) for m in MOTOR_NAMES]
            sdk_vals = self._sdk_joints(dataset_vals)
            achieved, violations = self._apply_limits(sdk_vals, side)

            for violation in violations:
                self._clamp_counts[f"{prefix}_{violation.split(':')[0].split('_', 1)[1]}.pos"] += 1
                if self._frame % 30 == 0:
                    logger.warning(f"[SIM] CLAMPED  {violation}")

            # Store achieved state (un-swap back to dataset convention for state)
            if self.config.swap_joints_2_3:
                achieved_dataset = [achieved[0], achieved[2], achieved[1],
                                    achieved[3], achieved[4], achieved[5], achieved[6]]
            else:
                achieved_dataset = achieved

            for i, m in enumerate(MOTOR_NAMES):
                self._state[f"{prefix}_{m}.pos"] = achieved_dataset[i]

        if self._frame % 30 == 0:
            cmd_l = [f"{action[f'left_{m}.pos']:.1f}" for m in MOTOR_NAMES[:6]]
            act_l = [f"{self._state[f'left_{m}.pos']:.1f}" for m in MOTOR_NAMES[:6]]
            logger.info(f"[SIM] Frame {self._frame:4d} | CMD left joints: {cmd_l}")
            logger.info(f"[SIM] Frame {self._frame:4d} | ACT left joints: {act_l}")

        self._frame += 1
        return action

    def disconnect(self) -> None:
        self._is_connected = False
        total_clamps = sum(self._clamp_counts.values())
        logger.info(f"[SIM] Disconnected after {self._frame} frames | Total clamp events: {total_clamps}")
        if total_clamps > 0:
            logger.warning("[SIM] Joints that were clamped:")
            for k, v in self._clamp_counts.items():
                if v > 0:
                    logger.warning(f"  {k}: {v} frames clamped")
