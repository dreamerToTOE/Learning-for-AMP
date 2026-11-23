"""Evaluate a saved skrl policy on Isaac Lab v2.1.1 Humanoid AMP Walk."""

import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000, help="Seed for the first evaluation reset")
    parser.add_argument("--video", action="store_true", help="Render a side-view skeleton of the first episode")
    parser.add_argument(
        "--action-range-correction",
        action="store_true",
        help="Apply the opt-in local action clipping and half-range mapping used by the R1 correction smoke",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes < 1 or not 0 <= args.seed <= 2**31 - args.episodes:
        parser.error("expected positive episodes and non-negative 31-bit episode seeds")
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


def record_skeleton(body_names, poses, path, dt):
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
            cv2.putText(frame, f"t = {step * dt:.2f} s    +X >", (22, 338), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (220, 220, 220), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def evaluate(args, agent_config):
    import numpy as np
    import torch
    import yaml
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from scipy.stats import wasserstein_distance
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import gaussian_model

    if args.action_range_correction:
        from r1_action_range_patch import apply_action_range_patch

        apply_action_range_patch()

    class EvaluationWalkEnv(HumanoidAmpEnv):
        def _get_dones(self):
            # DirectRLEnv auto-resets before returning the terminal step.
            self.step_positions = self.robot.data.body_pos_w[0].clone()
            self.step_velocity = self.robot.data.body_lin_vel_w[0, self.ref_body_index].clone()
            return super()._get_dones()

    config = yaml.safe_load(agent_config.read_text())
    policy_cfg = config["models"]["policy"].copy()
    if policy_cfg.pop("class") != "GaussianMixin" or config["agent"]["state_preprocessor"] != "RunningStandardScaler":
        raise ValueError("expected a skrl Gaussian policy and RunningStandardScaler")

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env = EvaluationWalkEnv(env_cfg)
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

        episodes = []
        with torch.inference_mode():
            for index in range(args.episodes):
                seed = args.seed + index
                obs, _ = env.reset(seed=seed)
                initial_obs = obs["policy"].cpu().numpy().copy()
                initial_position = env.robot.data.body_pos_w[0, env.ref_body_index].clone()
                poses = [env.robot.data.body_pos_w[0].clone()] if args.video and index == 0 else None
                speeds = []
                for step in range(1, env.max_episode_length + 1):
                    actions, _, _ = policy.compute({"states": normalizer(obs["policy"])}, role="policy")
                    if not torch.isfinite(actions).all():
                        raise ValueError(f"non-finite action in episode {index + 1}, step {step}")
                    obs, _, terminated, truncated, _ = env.step(actions)
                    speeds.append(env.step_velocity[:2])
                    if poses is not None:
                        poses.append(env.step_positions)
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
                    )

                terminal_position = env.step_positions[env.ref_body_index]
                delta = (terminal_position - initial_position).cpu().tolist()
                elapsed = step * env.step_dt
                speeds = torch.stack(speeds)
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
                    "forward_dx_m": delta[0],
                    "net_forward_speed_mps": delta[0] / elapsed,
                    "mean_xy_speed_mps": speeds.norm(dim=1).mean().item(),
                    "torso_vx_mps": speeds[:, 0].cpu().tolist(),
                    "terminal_torso_height_m": terminal_position[2].item(),
                }
                print(json.dumps({key: value for key, value in result.items() if key != "torso_vx_mps"},
                                 allow_nan=False), flush=True)
                episodes.append(result)

        with np.load(env_cfg.motion_file, allow_pickle=False) as motion:
            torso = motion["body_names"].tolist().index(env_cfg.reference_body)
            reference_vx = motion["body_linear_velocities"][:, torso, 0]
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
        (args.output / "episodes.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
