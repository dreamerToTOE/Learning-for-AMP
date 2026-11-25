"""Evaluate a saved skrl policy on Isaac Lab v2.1.1 Humanoid AMP Walk."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000, help="Seed for the first evaluation reset")
    parser.add_argument("--target-vx", type=float, default=None, help="Optional fixed world +X speed target in m/s")
    parser.add_argument("--video", action="store_true", help="Render a side-view skeleton of the first episode")
    parser.add_argument("--render-video", action="store_true", help="Record the Isaac Sim viewport of the first episode")
    parser.add_argument("--motion-metrics", action="store_true", help="Record independent gait and tracking metrics")
    parser.add_argument(
        "--action-range-correction",
        action="store_true",
        help="Apply the opt-in local action clipping and half-range mapping used by the R1 correction smoke",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.seed <= 2**31 - args.episodes:
        parser.error("expected positive episodes and non-negative 31-bit episode seeds")
    if args.target_vx is not None and not math.isfinite(args.target_vx):
        parser.error("--target-vx must be finite")
    if args.motion_metrics and (args.target_vx is None or args.target_vx <= 0):
        parser.error("--motion-metrics requires a positive --target-vx")
    if args.render_video:
        args.enable_cameras = True
    args.checkpoint = args.checkpoint.resolve(strict=True)
    agent_config = args.checkpoint.parents[1] / "params" / "agent.yaml"
    if not agent_config.is_file():
        parser.error(f"missing saved agent configuration: {agent_config}")
    args.output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    try:
        evaluate(args, agent_config)
    finally:
        app.close()


def record_skeleton(body_names, poses, path, dt, follow_torso=False):
    import cv2
    import numpy as np

    chains = [
        ("pelvis", "torso", "head"),
        ("torso", "right_upper_arm", "right_lower_arm", "right_hand"),
        ("torso", "left_upper_arm", "left_lower_arm", "left_hand"),
        ("pelvis", "right_thigh", "right_shin", "right_foot"),
        ("pelvis", "left_thigh", "left_shin", "left_foot"),
    ]
    chains = [[body_names.index(name) for name in chain] for chain in chains]
    poses = np.asarray(poses)[..., [0, 2]]
    if not np.isfinite(poses).all():
        raise ValueError("non-finite skeleton coordinates")
    travel = None
    if follow_torso:
        torso_x = poses[:, body_names.index("torso"), 0].copy()
        travel = torso_x - torso_x[0]
        poses[..., 0] -= torso_x[:, None]
    scale = min(480 / max(float(np.ptp(poses[..., 0])), 1), 250 / max(float(poses[..., 1].max()), 1))
    x_center = (poses[..., 0].max() + poses[..., 0].min()) / 2
    pixels = np.rint((poses - [x_center, 0]) * [scale, -scale] + [320, 315]).astype(np.int32)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / dt, (640, 360))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        for step, pose in enumerate(pixels):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.line(frame, (20, 315), (620, 315), (140, 140, 140), 1, cv2.LINE_AA)
            for chain, color in zip(chains, [(235, 235, 235), (185, 135, 75), (95, 200, 240)] * 2):
                cv2.polylines(frame, [pose[chain]], False, color, 4, cv2.LINE_AA)
            label = f"t = {step * dt:.2f} s    +X >"
            if travel is not None:
                label = f"t = {step * dt:.2f} s    torso X: {travel[step]:+.2f} m    +X >"
            cv2.putText(frame, label, (22, 338), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def record_viewport_frame(env, writer, position):
    import cv2
    from pxr import Gf

    x, y = position[:2].tolist()
    eye = Gf.Vec3d(x + 1.2, y - 3.2, 1.55)
    target = Gf.Vec3d(x, y, 0.85)
    view = Gf.Matrix4d(1.0).SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    env.render_camera_transform.Set(view.GetInverse())
    env.sim.render()
    rgb = env.render(recompute=True)
    if rgb.any():
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def evaluate(args, agent_config):
    import numpy as np
    import torch
    import yaml
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from scipy.stats import wasserstein_distance
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import gaussian_model

    if args.motion_metrics:
        from r1_walk_motion_metrics import (
            FOOT_NAMES,
            POSTURE_BODY_NAMES,
            episode_motion,
            is_complete_forward_motion,
            reference_motion,
            upright_posture,
            walking_windows,
        )

    if args.action_range_correction:
        from r1_action_range_patch import apply_action_range_patch

        apply_action_range_patch()

    class EvaluationWalkEnv(HumanoidAmpEnv):
        def _get_dones(self):
            # DirectRLEnv auto-resets before returning the terminal step.
            self.step_positions = self.robot.data.body_pos_w[0].clone()
            self.step_velocity = self.robot.data.body_lin_vel_w[0, self.ref_body_index].clone()
            if args.motion_metrics:
                self.step_joint_state = torch.stack((self.robot.data.joint_pos[0], self.robot.data.joint_vel[0]))
                self.step_torso_quaternion = self.robot.data.body_quat_w[0, self.ref_body_index].clone()
            if self.viewport_writer is not None:
                record_viewport_frame(self, self.viewport_writer, self.step_positions[self.ref_body_index])
            return super()._get_dones()

        viewport_writer = None

    config = yaml.safe_load(agent_config.read_text())
    policy_cfg = config["models"]["policy"].copy()
    if policy_cfg.pop("class") != "GaussianMixin" or config["agent"]["state_preprocessor"] != "RunningStandardScaler":
        raise ValueError("expected a skrl Gaussian policy and RunningStandardScaler")

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    if args.render_video:
        env_cfg.viewer.resolution = (640, 360)
        env_cfg.viewer.cam_prim_path = "/World/R1EvaluationCamera"
    env = EvaluationWalkEnv(env_cfg, render_mode="rgb_array" if args.render_video else None)
    viewport_writer = None
    try:
        obs_space = env.single_observation_space["policy"]
        policy = gaussian_model(
            observation_space=obs_space, action_space=env.single_action_space, device=env.device, **policy_cfg
        )
        policy.init_state_dict("policy")
        normalizer = RunningStandardScaler(
            size=obs_space, device=env.device, **(config["agent"].get("state_preprocessor_kwargs") or {})
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(checkpoint["policy"])
        normalizer.load_state_dict(checkpoint["state_preprocessor"])
        policy.eval()
        normalizer.eval()

        joint_names = list(env.robot.data.joint_names)
        with np.load(env_cfg.motion_file, allow_pickle=False) as motion:
            torso = motion["body_names"].tolist().index(env_cfg.reference_body)
            reference_vx = motion["body_linear_velocities"][:, torso, 0]
            if args.motion_metrics:
                reference = reference_motion(motion, joint_names, env.step_dt)
                foot_indices = [env.robot.data.body_names.index(name) for name in FOOT_NAMES]
                posture_indices = [env.robot.data.body_names.index(name) for name in POSTURE_BODY_NAMES]

        if args.render_video:
            import cv2
            from pxr import UsdGeom

            camera = UsdGeom.Camera.Define(env.sim.stage, env_cfg.viewer.cam_prim_path)
            camera.CreateFocalLengthAttr(18.0)
            env.render_camera_transform = UsdGeom.Xformable(camera.GetPrim()).AddTransformOp()
            viewport_writer = cv2.VideoWriter(
                str(args.output / "viewport_001.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                1 / env.step_dt, env_cfg.viewer.resolution,
            )
            if not viewport_writer.isOpened():
                raise RuntimeError("could not open the viewport video")
            env.viewport_writer = viewport_writer

        # HumanoidAmpEnv ends an episode at max_episode_length - 1.
        horizon_steps = env.max_episode_length - 1
        episodes = []
        with torch.inference_mode():
            for index in range(args.episodes):
                seed = args.seed + index
                obs, _ = env.reset(seed=seed)
                initial_obs = obs["policy"].cpu().numpy().copy()
                initial_position = env.robot.data.body_pos_w[0, env.ref_body_index].clone()
                poses = [env.robot.data.body_pos_w[0].clone()] if args.video and index == 0 else None
                if env.viewport_writer is not None:
                    # Initialize the annotator and draw the initial pose before the first action.
                    env.render()
                    record_viewport_frame(env, viewport_writer, initial_position)
                speeds = []
                if args.motion_metrics:
                    joint_states = []
                    action_samples = []
                    foot_positions = []
                    torso_x = [initial_position[0]]
                    body_heights = []
                    torso_quaternions = []
                target_violation_count = torch.zeros(env.robot.num_joints, device=env.device)
                target_error_sum = 0.0
                target_error_max = 0.0
                target_sample_count = 0
                for step in range(1, env.max_episode_length + 1):
                    actions, _, _ = policy.compute({"states": normalizer(obs["policy"])}, role="policy")
                    if not torch.isfinite(actions).all():
                        raise ValueError(f"non-finite action in episode {index + 1}, step {step}")
                    target = env.action_offset + env.action_scale * actions
                    limits = env.robot.data.soft_joint_pos_limits[0]
                    target_violation_count += ((target[0] < limits[:, 0]) | (target[0] > limits[:, 1])).float()
                    target_error = (env.robot.data.joint_pos[0] - target[0]).abs()
                    target_error_sum += target_error.mean().item()
                    target_error_max = max(target_error_max, target_error.max().item())
                    target_sample_count += 1
                    obs, _, terminated, truncated, _ = env.step(actions)
                    speeds.append(env.step_velocity[:2])
                    if poses is not None:
                        poses.append(env.step_positions)
                    if args.motion_metrics:
                        joint_states.append(env.step_joint_state)
                        action_samples.append(actions[0])
                        foot_positions.append(env.step_positions[foot_indices])
                        torso_x.append(env.step_positions[env.ref_body_index, 0])
                        body_heights.append(env.step_positions[posture_indices, 2])
                        torso_quaternions.append(env.step_torso_quaternion)
                    if terminated.item() or truncated.item():
                        break
                else:
                    raise RuntimeError("episode did not terminate within the configured horizon")
                if poses is not None:
                    record_skeleton(
                        env.robot.data.body_names,
                        torch.stack(poses).cpu().numpy(),
                        args.output / "episode_001.mp4",
                        env.step_dt,
                        follow_torso=args.motion_metrics,
                    )
                env.viewport_writer = None

                terminal_position = env.step_positions[env.ref_body_index]
                delta = (terminal_position - initial_position).cpu().tolist()
                elapsed = step * env.step_dt
                speeds = torch.stack(speeds)
                torso_vx = speeds[:, 0].cpu().numpy()
                result = {
                    "episode": index + 1,
                    "seed": seed,
                    "initial_observation_sha256": hashlib.sha256(initial_obs.tobytes()).hexdigest(),
                    "initial_position_m": initial_position.cpu().tolist(),
                    "steps": step,
                    "duration_s": elapsed,
                    "terminated": bool(terminated.item()),
                    "truncated": bool(truncated.item()),
                    "end_reason": "fall" if terminated.item() else "timeout",
                    "failure_reason": "torso_height_below_0.5m" if terminated.item() else "timeout_10s",
                    "forward_dx_m": delta[0],
                    "net_forward_speed_mps": delta[0] / elapsed,
                    "mean_xy_speed_mps": speeds.norm(dim=1).mean().item(),
                    "torso_vx_mps": torso_vx.tolist(),
                    "terminal_torso_height_m": terminal_position[2].item(),
                    "target_angle_violation_ratio": float(target_violation_count.sum().item() / max(target_sample_count * env.robot.num_joints, 1)),
                    "target_angle_violation_by_joint": {
                        name: float(value / max(target_sample_count, 1))
                        for name, value in zip(joint_names, target_violation_count.cpu().tolist())
                    },
                    "target_position_error_mean_rad": float(target_error_sum / max(target_sample_count, 1)),
                    "target_position_error_max_rad": float(target_error_max),
                }
                if args.target_vx is not None:
                    speed_error = torso_vx - args.target_vx
                    result["target_vx_mps"] = args.target_vx
                    result["target_vx_mae_mps"] = float(abs(speed_error).mean())
                    result["target_vx_bias_mps"] = float(speed_error.mean())
                    result["target_vx_within_0_2_ratio"] = float((abs(speed_error) < 0.2).mean())
                if args.motion_metrics:
                    if step > horizon_steps or (bool(truncated.item()) != (step == horizon_steps)):
                        raise RuntimeError("unexpected time limit for the fixed-horizon metrics")
                    joint_pos, joint_vel = torch.stack(joint_states, dim=1).cpu().numpy()
                    result["motion"] = episode_motion(
                        joint_pos, joint_vel, torch.stack(action_samples).cpu().numpy(),
                        torch.stack(foot_positions).cpu().numpy(), reference, joint_names, env.step_dt,
                    )
                    windows = walking_windows(torch.stack(torso_x).cpu().numpy(), env.step_dt)
                    complete = result["truncated"] and not result["terminated"]
                    result["motion"]["forward_window_speed_mps"] = windows
                    result["complete_forward_motion"] = is_complete_forward_motion(
                        complete, windows, result["motion"]["gait"]["forward_step_frames"], step, args.target_vx,
                    )
                    posture = upright_posture(
                        torch.stack(body_heights).cpu().numpy(), torch.stack(torso_quaternions).cpu().numpy(),
                    )
                    result["motion"]["posture"] = posture
                    result["posture_samples_available"] = True
                    result["complete_walk"] = (
                        result["complete_forward_motion"] and posture["terminal_upright"]
                        and posture["upright_fraction"] >= 0.95
                    )
                    result["target_vx_fixed_horizon_mae_mps"] = float(
                        (abs(speed_error).sum() + (horizon_steps - step) * args.target_vx) / horizon_steps
                    )
                    result["reference_vx_wasserstein_mps"] = float(wasserstein_distance(torso_vx, reference_vx))
                print(json.dumps({key: value for key, value in result.items() if key != "torso_vx_mps"},
                                 allow_nan=False), flush=True)
                episodes.append(result)

        rollout_vx = np.concatenate([episode["torso_vx_mps"] for episode in episodes])
        with args.checkpoint.open("rb") as checkpoint_file:
            checkpoint_sha256 = hashlib.sha256(checkpoint_file.read()).hexdigest()
        result = {
            "task": task,
            "algorithm": config["agent"]["class"],
            "checkpoint_sha256": checkpoint_sha256,
            "training_seed": config["seed"],
            "step_dt_s": env.step_dt,
            "reference_vx_wasserstein_mps": float(wasserstein_distance(rollout_vx, reference_vx)),
            "episodes": episodes,
        }
        if args.target_vx is not None:
            target_error = rollout_vx - args.target_vx
            result["target_vx_mps"] = args.target_vx
            result["target_vx_mae_mps"] = float(abs(target_error).mean())
            result["target_vx_bias_mps"] = float(target_error.mean())
            result["target_vx_within_0_2_ratio"] = float((abs(target_error) < 0.2).mean())
        if args.motion_metrics:
            result["horizon_steps"] = horizon_steps
            result["reference_motion"] = reference[2]
            result["reference_motion"]["sha256"] = hashlib.sha256(Path(env_cfg.motion_file).read_bytes()).hexdigest()
        (args.output / "episodes.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        if viewport_writer is not None:
            viewport_writer.release()
        env.close()


if __name__ == "__main__":
    main()
