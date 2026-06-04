from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    # CAN interface names for left and right arms
    left_can: str = "can1"
    right_can: str = "can0"

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "left_gripper": RealSenseCameraConfig(
                serial_number_or_name="233622071845",
                width=640,
                height=480,
                fps=30,
            ),
            "right_gripper": RealSenseCameraConfig(
                serial_number_or_name="233622073222",
                width=640,
                height=480,
                fps=30,
            ),
            "head": RealSenseCameraConfig(
                serial_number_or_name="938422072905",
                width=640,
                height=480,
                fps=30,
            ),
        }
    )
