"""Probe Humanoid28 actuator tensors at the first gravity replay action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


JOINT_NAMES = [
    "abdomen_x", "abdomen_y", "abdomen_z", "neck_x", "neck_y", "neck_z",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z",
    "right_hip_x", "right_hip_y", "right_hip_z", "left_hip_x", "left_hip_y", "left_hip_z",
    "right_elbow", "left_elbow", "right_knee", "left_knee",
    "right_ankle_x", "right_ankle_y", "right_ankle_z", "left_ankle_x", "left_ankle_y", "left_ankle_z",
]


USD_STIFFNESS = [
    34377.46875, 34377.46875, 34377.46875, 2864.7890625, 2864.7890625, 2864.7890625,
    11459.15625, 11459.15625, 11459.15625, 11459.15625, 11459.15625, 11459.15625,
    17188.734375, 17188.734375, 17188.734375, 17188.734375, 17188.734375, 17188.734375,
    8594.3671875, 8594.3671875, 17188.734375, 17188.734375,
    11459.15625, 11459.15625, 11459.15625, 11459.15625, 11459.15625, 11459.15625,
]


USD_DAMPING = [
    3437.74675, 3437.74675, 3437.74675, 286.4789, 286.4789, 286.4789,
    1145.91565, 1145.91565, 1145.91565, 1145.91565, 1145.91565, 1145.91565,
    1718.8734, 1718.8734, 1718.8734, 1718.8734, 1718.8734, 1718.8734,
    859.4367, 859.4367, 1718.8734, 1718.8734,
    1145.91565, 1145.91565, 1145.91565, 1145.91565, 1145.91565, 1145.91565,
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--ground-friction", type=float, default=0.5)
    parser.add_argument("--override-usd-drives", action="store_true")
    parser.add_argument("--repair-usd-drives", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        probe(args)
    finally:
        app.close()


def probe(args) -> None:
    import numpy as np
    import torch
    from isaaclab_tasks.direct.humanoid_amp import humanoid_amp_env as amp_env_module
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    if args.override_usd_drives:
        from r1_humanoid28_drive_patch import apply_humanoid28_drive_override

    original_ground_plane_cfg = amp_env_module.GroundPlaneCfg

    def patched_ground_plane_cfg(*ground_args, **ground_kwargs):
        cfg = original_ground_plane_cfg(*ground_args, **ground_kwargs)
        cfg.physics_material.static_friction = args.ground_friction
        cfg.physics_material.dynamic_friction = args.ground_friction
        return cfg

    amp_env_module.GroundPlaneCfg = patched_ground_plane_cfg
    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.reset_strategy = "random-start"
    if args.repair_usd_drives:
        actuator_cfg = env_cfg.robot.actuators["body"]
        actuator_cfg.stiffness = dict(zip(JOINT_NAMES, USD_STIFFNESS))
        actuator_cfg.damping = dict(zip(JOINT_NAMES, USD_DAMPING))
        actuator_cfg.effort_limit_sim = 1.0e9
        actuator_cfg.velocity_limit_sim = 1.0e9
    if args.override_usd_drives:
        apply_humanoid28_drive_override(env_cfg)
    env = HumanoidAmpEnv(env_cfg)
    try:
        env.reset(seed=args.seed)
        motion = np.load(env_cfg.motion_file, allow_pickle=False)
        dof_names = motion["dof_names"].tolist()
        index = [dof_names.index(name) for name in env.robot.data.joint_names]
        frame = args.start_frame
        target_frame = min(frame + int(env.cfg.decimation), len(motion["dof_positions"]) - 1)
        body_names = motion["body_names"].tolist()
        torso_index = body_names.index(env_cfg.reference_body)
        root_state = env.robot.data.root_state_w[0].clone()
        root_state[:3] = torch.as_tensor(motion["body_positions"][frame, torso_index], device=env.device)
        root_state[:3] += env.scene.env_origins[0]
        root_state[3:7] = torch.as_tensor(motion["body_rotations"][frame, torso_index], device=env.device)
        root_state[7:10] = torch.as_tensor(motion["body_linear_velocities"][frame, torso_index], device=env.device)
        root_state[10:13] = torch.as_tensor(motion["body_angular_velocities"][frame, torso_index], device=env.device)
        dof_positions = torch.as_tensor(motion["dof_positions"][:, index], device=env.device)
        joint_velocity = torch.as_tensor(motion["dof_velocities"][frame, index], device=env.device)
        env.robot.write_root_link_pose_to_sim(root_state[:7].unsqueeze(0), env.robot._ALL_INDICES)
        env.robot.write_root_com_velocity_to_sim(root_state[7:].unsqueeze(0), env.robot._ALL_INDICES)
        env.robot.write_joint_state_to_sim(dof_positions[frame].unsqueeze(0), joint_velocity.unsqueeze(0), None, env.robot._ALL_INDICES)
        target = dof_positions[target_frame]
        actions = ((target - env.action_offset) / env.action_scale).unsqueeze(0)
        env.step(actions)
        actuator = next(iter(env.robot.actuators.values()))
        names = list(env.robot.data.joint_names)
        values = {}
        for attr_name in ("stiffness", "damping", "effort_limit", "velocity_limit"):
            value = getattr(actuator, attr_name, None)
            values[attr_name] = value[0].detach().cpu().tolist() if value is not None else None
        data = env.robot.data
        result = {
            "task": task,
            "device": str(args.device),
            "seed": args.seed,
            "start_frame": frame,
            "target_frame": target_frame,
            "ground_friction": args.ground_friction,
            "override_usd_drives": args.override_usd_drives,
            "repair_usd_drives": args.repair_usd_drives,
            "joint_names": names,
            "actuator_config": values,
            "joint_pos_target": data.joint_pos_target[0].detach().cpu().tolist(),
            "joint_vel_target": data.joint_vel_target[0].detach().cpu().tolist(),
            "joint_effort_target": data.joint_effort_target[0].detach().cpu().tolist(),
            "computed_torque": data.computed_torque[0].detach().cpu().tolist(),
            "applied_torque": data.applied_torque[0].detach().cpu().tolist(),
            "joint_pos": data.joint_pos[0].detach().cpu().tolist(),
            "joint_vel": data.joint_vel[0].detach().cpu().tolist(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
