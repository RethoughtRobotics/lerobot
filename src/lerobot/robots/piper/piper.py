import time
from functools import cached_property

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_piper import PiperRobotConfig

# SDK returns/expects joint angles in 0.001 deg units; dataset is in degrees.
JOINT_FACTOR = 1000  # 0.001 deg <-> deg
# SDK raw range ~0-10000; dataset range 0-100. Factor = 100.
GRIPPER_FACTOR = 100
HOME_POSITION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 6 joints (deg) + gripper (mm)
MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


class _PiperArm:
    """Thin wrapper around piper_sdk for a single arm."""

    def __init__(self, can_name: str):
        from piper_sdk import C_PiperInterface_V2

        self.sdk = C_PiperInterface_V2(can_name)
        self.sdk.ConnectPort()

    def enable(self, timeout: float = 5.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            self.sdk.EnableArm(7)
            self.sdk.GripperCtrl(0, 1000, 0x01, 0)
            statuses = [
                getattr(self.sdk.GetArmLowSpdInfoMsgs(), f"motor_{i}").foc_status.driver_enable_status
                for i in range(1, 7)
            ]
            if all(statuses):
                return True
            time.sleep(0.5)
        return False

    def disable(self):
        self.sdk.DisableArm(7)
        self.sdk.GripperCtrl(0, 1000, 0x02, 0)

    def home(self):
        self.write(HOME_POSITION)

    def read(self) -> dict[str, float]:
        j = self.sdk.GetArmJointMsgs().joint_state
        g = self.sdk.GetArmGripperMsgs().gripper_state
        return {
            "joint1": j.joint_1 / JOINT_FACTOR,
            "joint2": j.joint_2 / JOINT_FACTOR,
            "joint3": j.joint_3 / JOINT_FACTOR,
            "joint4": j.joint_4 / JOINT_FACTOR,
            "joint5": j.joint_5 / JOINT_FACTOR,
            "joint6": j.joint_6 / JOINT_FACTOR,
            "gripper": g.grippers_angle / GRIPPER_FACTOR,
        }

    def write(self, joints: list[float]):
        self.sdk.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.sdk.JointCtrl(
            round(joints[0] * JOINT_FACTOR),
            round(joints[1] * JOINT_FACTOR),
            round(joints[2] * JOINT_FACTOR),
            round(joints[3] * JOINT_FACTOR),
            round(joints[4] * JOINT_FACTOR),
            round(joints[5] * JOINT_FACTOR),
        )
        self.sdk.GripperCtrl(abs(round(joints[6] * GRIPPER_FACTOR)), 1000, 0x01, 0)


class PiperRobot(Robot):
    """
    Bimanual Agilex Piper robot (14-DOF: 2 × 7-DOF arms via CAN bus).
    Expects left arm on `left_can` and right arm on `right_can`.
    """

    config_class = PiperRobotConfig
    name = "piper"

    def __init__(self, config: PiperRobotConfig):
        super().__init__(config)
        self.config = config
        self._left: _PiperArm | None = None
        self._right: _PiperArm | None = None
        self._is_connected = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def observation_features(self) -> dict:
        motor_ft = {f"left_{m}.pos": float for m in MOTOR_NAMES}
        motor_ft.update({f"right_{m}.pos": float for m in MOTOR_NAMES})
        cam_ft = {
            name: (cfg.height, cfg.width, 3)
            for name, cfg in self.config.cameras.items()
        }
        return {**motor_ft, **cam_ft}

    @cached_property
    def action_features(self) -> dict:
        return {f"left_{m}.pos": float for m in MOTOR_NAMES} | {
            f"right_{m}.pos": float for m in MOTOR_NAMES
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._left = _PiperArm(self.config.left_can)
        self._right = _PiperArm(self.config.right_can)

        if not self._left.enable():
            raise RuntimeError(f"Left arm failed to enable on {self.config.left_can}")
        if not self._right.enable():
            raise RuntimeError(f"Right arm failed to enable on {self.config.right_can}")

        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True

        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        self._left.home()
        self._right.home()

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        left_state = self._left.read()
        right_state = self._right.read()

        obs: RobotObservation = {}
        for m in MOTOR_NAMES:
            obs[f"left_{m}.pos"] = left_state[m]
            obs[f"right_{m}.pos"] = right_state[m]

        for name, cam in self.cameras.items():
            obs[name] = cam.async_read()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_joints = [action[f"left_{m}.pos"] for m in MOTOR_NAMES]
        right_joints = [action[f"right_{m}.pos"] for m in MOTOR_NAMES]

        self._left.write(left_joints)
        self._right.write(right_joints)

        return action

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        if self._left:
            self._left.home()
            self._left.disable()
        if self._right:
            self._right.home()
            self._right.disable()
        for cam in self.cameras.values():
            cam.disconnect()
        self._is_connected = False
