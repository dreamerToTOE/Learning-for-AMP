"""Compare the Humanoid28 runtime state with one reference-motion frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--no-reset-lift", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        compare(args)
    finally:
        app.close()


def compare(args) -> None:
    import numpy as np
    import torch
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.reset_strategy = "random-start"
    env = HumanoidAmpEnv(env_cfg)
    try:
        obs, _ = env.reset(seed=args.seed)
        motion = np.load(env_cfg.motion_file, allow_pickle=False)
        if not 0 <= args.frame < len(motion["dof_positions"]):
            raise ValueError(f"frame out of range: {args.frame}")

        motion_body_names = motion["body_names"].tolist()
        motion_dof_names = motion["dof_names"].tolist()
        runtime_body_names = list(env.robot.data.body_names)
        runtime_dof_names = list(env.robot.data.joint_names)
        body_index = [motion_body_names.index(name) for name in runtime_body_names]
        dof_index = [motion_dof_names.index(name) for name in runtime_dof_names]

        expected_offset = torch.tensor([0.0, 0.0, 0.0 if args.no_reset_lift else 0.15], device=env.device)
        motion_body_pos = torch.as_tensor(motion["body_positions"][args.frame, body_index], device=env.device)
        expected_body_pos = motion_body_pos + expected_offset
        actual_body_pos = env.robot.data.body_pos_w[0]
        position_error = actual_body_pos - expected_body_pos

        motion_body_vel = torch.as_tensor(
            motion["body_linear_velocities"][args.frame, body_index], device=env.device
        )
        actual_body_vel = env.robot.data.body_lin_vel_w[0]
        velocity_error = actual_body_vel - motion_body_vel

        motion_joint_pos = torch.as_tensor(motion["dof_positions"][args.frame, dof_index], device=env.device)
        motion_joint_vel = torch.as_tensor(motion["dof_velocities"][args.frame, dof_index], device=env.device)
        joint_pos_error = env.robot.data.joint_pos[0] - motion_joint_pos
        joint_vel_error = env.robot.data.joint_vel[0] - motion_joint_vel

        limits = env.robot.data.soft_joint_pos_limits[0]
        motion_all_joints = torch.as_tensor(motion["dof_positions"][:, dof_index], device=env.device)
        out_of_limits = (motion_all_joints < limits[:, 0]) | (motion_all_joints > limits[:, 1])
        motion_min = motion_all_joints.min(dim=0).values
        motion_max = motion_all_joints.max(dim=0).values

        masses = env.robot.data.default_mass[0]
        inertias = env.robot.data.default_inertia[0]
        result = {
            "task": task,
            "seed": args.seed,
            "frame": args.frame,
            "reset_lift_m": 0.0 if args.no_reset_lift else 0.15,
            "runtime_body_names": runtime_body_names,
            "motion_body_names": motion_body_names,
            "runtime_dof_names": runtime_dof_names,
            "motion_dof_names": motion_dof_names,
            "body_position_error_max_m": position_error.norm(dim=-1).max().item(),
            "body_position_error_mean_m": position_error.norm(dim=-1).mean().item(),
            "body_position_error_by_body_m": position_error.norm(dim=-1).cpu().tolist(),
            "body_linear_velocity_error_max_mps": velocity_error.norm(dim=-1).max().item(),
            "body_linear_velocity_error_mean_mps": velocity_error.norm(dim=-1).mean().item(),
            "body_linear_velocity_error_by_body_mps": velocity_error.norm(dim=-1).cpu().tolist(),
            "runtime_body_linear_velocity_by_body_mps": actual_body_vel.cpu().tolist(),
            "motion_body_linear_velocity_by_body_mps": motion_body_vel.cpu().tolist(),
            "joint_position_error_max_rad": joint_pos_error.abs().max().item(),
            "joint_velocity_error_max_radps": joint_vel_error.abs().max().item(),
            "joint_position_error_by_joint_rad": joint_pos_error.cpu().tolist(),
            "joint_velocity_error_by_joint_radps": joint_vel_error.cpu().tolist(),
            "soft_limits_min_rad": limits[:, 0].cpu().tolist(),
            "soft_limits_max_rad": limits[:, 1].cpu().tolist(),
            "motion_min_rad": motion_min.cpu().tolist(),
            "motion_max_rad": motion_max.cpu().tolist(),
            "motion_out_of_soft_limits_count": int(out_of_limits.sum().item()),
            "motion_out_of_soft_limits_by_joint": out_of_limits.any(dim=0).cpu().tolist(),
            "default_mass_by_body_kg": masses.cpu().tolist(),
            "default_inertia_by_body": inertias.cpu().tolist(),
            "physics_dt_s": env.cfg.sim.dt,
            "decimation": env.cfg.decimation,
            "env_step_dt_s": env.step_dt,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
