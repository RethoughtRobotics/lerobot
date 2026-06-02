from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    # CAN interface names for left and right arms
    left_can: str = "can0"
    right_can: str = "can1"

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left_gripper": RealSenseCameraConfig(
                serial_number_or_name="<LEFT_GRIPPER_SERIAL>",
                width=640,
                height=480,
                fps=30,
            ),
            "right_gripper": RealSenseCameraConfig(
                serial_number_or_name="<RIGHT_GRIPPER_SERIAL>",
                width=640,
                height=480,
                fps=30,
            ),
            "head": RealSenseCameraConfig(
                serial_number_or_name="<HEAD_SERIAL>",
                width=640,
                height=480,
                fps=30,
            ),
        }
    )
