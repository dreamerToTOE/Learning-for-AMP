"""Replay the official Walk motion as joint-position targets in the task."""

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
    parser.add_argument("--steps", type=int, default=77)
    parser.add_argument("--half-range", action="store_true")
    parser.add_argument("--hold-initial", action="store_true")
    parser.add_argument("--no-reset-lift", action="store_true")
    parser.add_argument("--stiffness", type=float, default=None)
    parser.add_argument("--damping", type=float, default=None)
    parser.add_argument("--disable-gravity", action="store_true")
    parser.add_argument("--solver-position-iterations", type=int, default=None)
    parser.add_argument("--solver-velocity-iterations", type=int, default=None)
    parser.add_argument("--repair-usd-drives", action="store_true")
    parser.add_argument("--zero-root-velocity", action="store_true")
    parser.add_argument("--disable-self-collisions", action="store_true")
    parser.add_argument("--explicit-pd", action="store_true")
    parser.add_argument("--pd-stiffness", type=float, default=1000.0)
    parser.add_argument("--pd-damping", type=float, default=100.0)
    parser.add_argument("--pd-effort-limit", type=float, default=1000.0)
    parser.add_argument("--override-usd-drives", action="store_true")
    parser.add_argument("--drive-gain-scale", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--contact-forces", action="store_true")
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--ground-friction", type=float, default=None)
    parser.add_argument("--ground-restitution", type=float, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        replay(args)
    finally:
        app.close()


def replay(args) -> None:
    import numpy as np
    import torch
    from isaacsim.core.prims import RigidPrim
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    if args.override_usd_drives:
        from r1_humanoid28_drive_patch import apply_humanoid28_drive_override

    class ReplayEnv(HumanoidAmpEnv):
        def _get_dones(self):
            self.terminal_root = self.robot.data.body_pos_w[0, self.ref_body_index].clone()
            self.terminal_feet = self.robot.data.body_pos_w[0, self.key_body_indexes].clone()
            self.terminal_joint_pos = self.robot.data.joint_pos[0].clone()
            if getattr(self, "contact_view", None) is not None:
                forces = self.contact_view.get_net_contact_forces(dt=self.step_dt)
                self.terminal_contact_forces = torch.as_tensor(forces, device=self.device).clone()
            return super()._get_dones()

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.reset_strategy = "random-start"
    if args.disable_gravity:
        env_cfg.robot.spawn.rigid_props.disable_gravity = True
    if args.disable_self_collisions:
        env_cfg.robot.spawn.articulation_props.enabled_self_collisions = False
    if args.solver_position_iterations is not None:
        env_cfg.robot.spawn.articulation_props.solver_position_iteration_count = args.solver_position_iterations
    if args.solver_velocity_iterations is not None:
        env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = args.solver_velocity_iterations
    if args.repair_usd_drives:
        actuator_cfg = env_cfg.robot.actuators["body"]
        actuator_cfg.stiffness = dict(zip(JOINT_NAMES, USD_STIFFNESS))
        actuator_cfg.damping = dict(zip(JOINT_NAMES, USD_DAMPING))
        actuator_cfg.effort_limit_sim = 1.0e9
        actuator_cfg.velocity_limit_sim = 1.0e9
    if args.explicit_pd:
        from isaaclab.actuators import IdealPDActuatorCfg

        env_cfg.robot.actuators = {
            "body": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=args.pd_stiffness,
                damping=args.pd_damping,
                effort_limit=args.pd_effort_limit,
                effort_limit_sim=args.pd_effort_limit,
            )
        }
    if args.override_usd_drives:
        apply_humanoid28_drive_override(env_cfg, gain_scale=args.drive_gain_scale)
    if args.stiffness is not None or args.damping is not None:
        actuator_cfg = env_cfg.robot.actuators["body"]
        if args.stiffness is not None:
            actuator_cfg.stiffness = args.stiffness
        if args.damping is not None:
            actuator_cfg.damping = args.damping
    if args.ground_friction is not None or args.ground_restitution is not None:
        from isaaclab_tasks.direct.humanoid_amp import humanoid_amp_env as amp_env_module

        original_ground_plane_cfg = amp_env_module.GroundPlaneCfg

        def patched_ground_plane_cfg(*ground_args, **ground_kwargs):
            cfg = original_ground_plane_cfg(*ground_args, **ground_kwargs)
            if args.ground_friction is not None:
                cfg.physics_material.static_friction = args.ground_friction
                cfg.physics_material.dynamic_friction = args.ground_friction
            if args.ground_restitution is not None:
                cfg.physics_material.restitution = args.ground_restitution
            return cfg

        amp_env_module.GroundPlaneCfg = patched_ground_plane_cfg
    env = ReplayEnv(env_cfg)
    try:
        env.contact_view = None
        contact_view_error = None
        if args.contact_forces:
            try:
                env.contact_view = RigidPrim(
                    prim_paths_expr="/World/envs/env_0/Robot/.*_foot",
                    name="r1_humanoid_feet_contact_view",
                    track_contact_forces=True,
                )
                env.contact_view.initialize()
            except Exception as exc:
                contact_view_error = repr(exc)
                env.contact_view = None
        if args.half_range:
            env.action_scale = 0.5 * env.action_scale
        obs, _ = env.reset(seed=args.seed)
        if args.no_reset_lift or args.zero_root_velocity:
            root_state = env.robot.data.root_state_w[0].clone()
            if args.no_reset_lift:
                root_state[2] -= 0.15
            if args.zero_root_velocity:
                root_state[7:] = 0.0
            env.robot.write_root_link_pose_to_sim(root_state[:7].unsqueeze(0), env.robot._ALL_INDICES)
            env.robot.write_root_com_velocity_to_sim(root_state[7:].unsqueeze(0), env.robot._ALL_INDICES)
        motion = np.load(env_cfg.motion_file, allow_pickle=False)
        dof_names = motion["dof_names"].tolist()
        index = [dof_names.index(name) for name in env.robot.data.joint_names]
        dof_positions = torch.as_tensor(motion["dof_positions"][:, index], device=env.device)
        if args.start_frame is not None:
            if not 0 <= args.start_frame < dof_positions.shape[0]:
                raise ValueError(f"start frame out of range: {args.start_frame}")
            body_names_motion = motion["body_names"].tolist()
            torso_index = body_names_motion.index(env_cfg.reference_body)
            root_state = env.robot.data.root_state_w[0].clone()
            root_state[:3] = torch.as_tensor(
                motion["body_positions"][args.start_frame, torso_index], device=env.device
            ) + env.scene.env_origins[0]
            root_state[2] += 0.0 if args.no_reset_lift else 0.15
            root_state[3:7] = torch.as_tensor(
                motion["body_rotations"][args.start_frame, torso_index], device=env.device
            )
            root_state[7:10] = torch.as_tensor(
                motion["body_linear_velocities"][args.start_frame, torso_index], device=env.device
            )
            root_state[10:13] = torch.as_tensor(
                motion["body_angular_velocities"][args.start_frame, torso_index], device=env.device
            )
            env.robot.write_root_link_pose_to_sim(root_state[:7].unsqueeze(0), env.robot._ALL_INDICES)
            env.robot.write_root_com_velocity_to_sim(root_state[7:].unsqueeze(0), env.robot._ALL_INDICES)
            joint_velocity = torch.as_tensor(
                motion["dof_velocities"][args.start_frame, index], device=env.device
            )
            env.robot.write_joint_state_to_sim(
                dof_positions[args.start_frame].unsqueeze(0), joint_velocity.unsqueeze(0), None, env.robot._ALL_INDICES
            )
        actuator = next(iter(env.robot.actuators.values()))
        joint_names = list(env.robot.data.joint_names)
        body_names = list(env.robot.data.body_names)
        actuator_summary = {}
        for name in ("stiffness", "damping", "effort_limit", "velocity_limit"):
            value = getattr(actuator, name, None)
            if value is not None:
                actuator_summary[name] = {
                    "min_max": [value.min().item(), value.max().item()],
                    "values": value.reshape(-1).cpu().tolist(),
                }
        root_initial = env.robot.data.body_pos_w[0, env.ref_body_index].clone()
        feet_initial = env.robot.data.body_pos_w[0, env.key_body_indexes].clone()
        initial_joint_error = (dof_positions[0] - env.robot.data.joint_pos[0]).cpu().tolist()
        steps = []
        with torch.inference_mode():
            for step in range(args.steps):
                stride = int(env.cfg.decimation) if args.frame_stride is None else args.frame_stride
                phase_offset = 0 if args.start_frame is None else args.start_frame
                frame = min(phase_offset + (step + 1) * stride, dof_positions.shape[0] - 1)
                target = dof_positions[0] if args.hold_initial else dof_positions[frame]
                actions = ((target - env.action_offset) / env.action_scale).unsqueeze(0)
                obs, _, terminated, truncated, _ = env.step(actions)
                steps.append(
                    {
                        "step": step,
                        "motion_frame": frame,
                        "action_min_max": [actions.min().item(), actions.max().item()],
                        "target_delta_max_abs": (target - env.terminal_joint_pos).abs().max().item(),
                        "target_delta_by_joint": (target - env.terminal_joint_pos).cpu().tolist(),
                        "root_terminal": env.terminal_root.cpu().tolist(),
                        "feet_terminal": env.terminal_feet.cpu().tolist(),
                        "contact_forces": (
                            env.terminal_contact_forces.cpu().tolist()
                            if args.contact_forces and hasattr(env, "terminal_contact_forces")
                            else None
                        ),
                        "contact_force_norms": (
                            env.terminal_contact_forces.norm(dim=-1).cpu().tolist()
                            if args.contact_forces and hasattr(env, "terminal_contact_forces")
                            else None
                        ),
                        "terminated": bool(terminated.item()),
                        "truncated": bool(truncated.item()),
                    }
                )
                if bool(terminated.item() or truncated.item()):
                    break
        result = {
            "task": task,
            "seed": args.seed,
            "half_range": args.half_range,
            "hold_initial": args.hold_initial,
            "no_reset_lift": args.no_reset_lift,
            "stiffness": args.stiffness,
            "damping": args.damping,
            "disable_gravity": args.disable_gravity,
            "solver_position_iterations": args.solver_position_iterations,
            "solver_velocity_iterations": args.solver_velocity_iterations,
            "repair_usd_drives": args.repair_usd_drives,
            "zero_root_velocity": args.zero_root_velocity,
            "disable_self_collisions": args.disable_self_collisions,
            "explicit_pd": args.explicit_pd,
            "pd_stiffness": args.pd_stiffness,
            "pd_damping": args.pd_damping,
            "pd_effort_limit": args.pd_effort_limit,
            "override_usd_drives": args.override_usd_drives,
            "drive_gain_scale": args.drive_gain_scale,
            "frame_stride": args.frame_stride,
            "contact_forces": args.contact_forces,
            "contact_view_error": contact_view_error,
            "start_frame": args.start_frame,
            "ground_friction": args.ground_friction,
            "ground_restitution": args.ground_restitution,
            "steps_requested": args.steps,
            "steps_completed": len(steps),
            "terminated_step": next((item["step"] for item in steps if item["terminated"]), None),
            "initial_root": root_initial.cpu().tolist(),
            "initial_feet": feet_initial.cpu().tolist(),
            "initial_joint_error": initial_joint_error,
            "joint_names": joint_names,
            "body_names": body_names,
            "asset_usd_path": env.robot.cfg.spawn.usd_path,
            "actuator_summary": actuator_summary,
            "steps": steps,
            "step_dt_s": env.step_dt,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
