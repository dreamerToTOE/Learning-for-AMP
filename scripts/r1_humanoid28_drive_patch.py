"""Project-local Humanoid28 drive override for Isaac Lab v2.1.1."""

from __future__ import annotations

import math


_JOINT_GROUPS = {
    "abdomen_.*": (600.0, 60.0),
    "neck_.*": (50.0, 5.0),
    "right_shoulder_.*": (200.0, 20.0),
    "left_shoulder_.*": (200.0, 20.0),
    "right_elbow": (150.0, 15.0),
    "left_elbow": (150.0, 15.0),
    "right_hip_.*": (300.0, 30.0),
    "left_hip_.*": (300.0, 30.0),
    "right_knee": (300.0, 30.0),
    "left_knee": (300.0, 30.0),
    "right_ankle_.*": (200.0, 20.0),
    "left_ankle_.*": (200.0, 20.0),
}


def apply_humanoid28_drive_override(env_cfg, gain_scale: float = 1.0) -> None:
    """Set explicit radian-space PD gains and nonzero solver limits.

    The Humanoid28 USD stores drive gains in degree-based joint prims. Isaac
    Lab's default ``None`` resolution leaves four revolute DOFs at zero in the
    runtime PhysX tensor. This override converts the known USD values to the
    runtime radian-space values and sets them explicitly by joint regex.
    """

    actuator_cfg = env_cfg.robot.actuators["body"]
    actuator_cfg.stiffness = {key: stiffness * 180.0 / math.pi * gain_scale for key, (stiffness, _) in _JOINT_GROUPS.items()}
    actuator_cfg.damping = {key: damping * 180.0 / math.pi * gain_scale for key, (_, damping) in _JOINT_GROUPS.items()}
    actuator_cfg.effort_limit_sim = {".*": 1.0e9}
    actuator_cfg.velocity_limit_sim = {".*": 1.0e6}
