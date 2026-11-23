"""Replay the official Walk motion as joint-position targets in the task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


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
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    class ReplayEnv(HumanoidAmpEnv):
        def _get_dones(self):
            self.terminal_root = self.robot.data.body_pos_w[0, self.ref_body_index].clone()
            self.terminal_feet = self.robot.data.body_pos_w[0, self.key_body_indexes].clone()
            self.terminal_joint_pos = self.robot.data.joint_pos[0].clone()
            return super()._get_dones()

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.reset_strategy = "random-start"
    if args.stiffness is not None or args.damping is not None:
        actuator_cfg = env_cfg.robot.actuators["body"]
        if args.stiffness is not None:
            actuator_cfg.stiffness = args.stiffness
        if args.damping is not None:
            actuator_cfg.damping = args.damping
    env = ReplayEnv(env_cfg)
    try:
        if args.half_range:
            env.action_scale = 0.5 * env.action_scale
        obs, _ = env.reset(seed=args.seed)
        if args.no_reset_lift:
            root_state = env.robot.data.root_state_w[0].clone()
            root_state[2] -= 0.15
            env.robot.write_root_link_pose_to_sim(root_state[:7].unsqueeze(0), env.robot._ALL_INDICES)
            env.robot.write_root_com_velocity_to_sim(root_state[7:].unsqueeze(0), env.robot._ALL_INDICES)
        motion = np.load(env_cfg.motion_file, allow_pickle=False)
        dof_names = motion["dof_names"].tolist()
        index = [dof_names.index(name) for name in env.robot.data.joint_names]
        dof_positions = torch.as_tensor(motion["dof_positions"][:, index], device=env.device)
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
        initial_joint_error = (dof_positions[0] - env.robot.data.joint_pos[0]).cpu().tolist()
        steps = []
        with torch.inference_mode():
            for step in range(args.steps):
                frame = min((step + 1) * int(env.cfg.decimation), dof_positions.shape[0] - 1)
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
            "steps_requested": args.steps,
            "steps_completed": len(steps),
            "terminated_step": next((item["step"] for item in steps if item["terminated"]), None),
            "initial_root": root_initial.cpu().tolist(),
            "initial_joint_error": initial_joint_error,
            "joint_names": joint_names,
            "body_names": body_names,
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
