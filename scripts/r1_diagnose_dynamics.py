"""One-step reset/action/contact probe for the official Humanoid AMP Walk task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--clip-actions", action="store_true")
    parser.add_argument("--half-range", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        probe(args)
    finally:
        app.close()


def probe(args):
    import torch
    import yaml
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from skrl.utils.model_instantiators.torch import deterministic_model, gaussian_model

    class ProbeEnv(HumanoidAmpEnv):
        def _get_dones(self):
            self.terminal_root = self.robot.data.body_pos_w[0, self.ref_body_index].clone()
            self.terminal_feet = self.robot.data.body_pos_w[0, self.key_body_indexes].clone()
            self.terminal_joint_pos = self.robot.data.joint_pos[0].clone()
            return super()._get_dones()

    config_path = args.checkpoint.resolve().parents[1] / "params" / "agent.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    env_cfg = parse_env_cfg("Isaac-Humanoid-AMP-Walk-Direct-v0", device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env = ProbeEnv(env_cfg)
    try:
        if args.half_range:
            env.action_scale = 0.5 * env.action_scale
        obs, _ = env.reset(seed=args.seed)
        policy_cfg = cfg["models"]["policy"].copy()
        if policy_cfg.pop("class") != "GaussianMixin":
            raise ValueError("expected GaussianMixin policy")
        policy = gaussian_model(
            observation_space=env.single_observation_space["policy"],
            action_space=env.single_action_space,
            device=env.device,
            **policy_cfg,
        )
        policy.init_state_dict("policy")
        normalizer = RunningStandardScaler(
            size=env.single_observation_space["policy"],
            device=env.device,
            **(cfg["agent"].get("state_preprocessor_kwargs") or {}),
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        policy.load_state_dict(checkpoint["policy"])
        normalizer.load_state_dict(checkpoint["state_preprocessor"])
        discriminator_cfg = cfg["models"]["discriminator"].copy()
        if discriminator_cfg.pop("class") != "DeterministicMixin":
            raise ValueError("expected DeterministicMixin discriminator")
        discriminator = deterministic_model(
            observation_space=env.amp_observation_space,
            action_space=env.single_action_space,
            device=env.device,
            **discriminator_cfg,
        )
        discriminator.load_state_dict(checkpoint["discriminator"])
        amp_normalizer = RunningStandardScaler(
            size=env.amp_observation_space,
            device=env.device,
            **(cfg["agent"].get("amp_state_preprocessor_kwargs") or {}),
        )
        amp_normalizer.load_state_dict(checkpoint["amp_state_preprocessor"])
        policy.eval()
        normalizer.eval()
        discriminator.eval()
        amp_normalizer.eval()
        with torch.inference_mode():
            initial_amp = env.extras["amp_obs"]
            initial_logit, _, _ = discriminator.act(
                {"states": amp_normalizer(initial_amp)}, role="discriminator"
            )
            initial_style_reward = -torch.log(
                torch.maximum(1 - torch.sigmoid(initial_logit), torch.tensor(0.0001, device=env.device))
            ) * float(cfg["agent"]["discriminator_reward_scale"])
        steps = []
        with torch.inference_mode():
            for step_idx in range(args.steps):
                actions, _, _ = policy.compute({"states": normalizer(obs["policy"])}, role="policy")
                applied_actions = torch.clamp(actions, -1.0, 1.0) if args.clip_actions else actions
                target = env.action_offset + env.action_scale * applied_actions[0]
                before_joint = env.robot.data.joint_pos[0].clone()
                before_root = env.robot.data.body_pos_w[0, env.ref_body_index].clone()
                before_feet = env.robot.data.body_pos_w[0, env.key_body_indexes].clone()
                obs, _, terminated, truncated, _ = env.step(applied_actions)
                steps.append(
                    {
                        "step": step_idx,
                        "raw_action_min_max": [actions.min().item(), actions.max().item()],
                        "applied_action_min_max": [applied_actions.min().item(), applied_actions.max().item()],
                        "target_joint_min_max": [target.min().item(), target.max().item()],
                        "target_delta_max_abs": (target - before_joint).abs().max().item(),
                        "root_before": before_root.cpu().tolist(),
                        "root_terminal": env.terminal_root.cpu().tolist(),
                        "feet_before": before_feet.cpu().tolist(),
                        "feet_terminal": env.terminal_feet.cpu().tolist(),
                        "joint_terminal_delta_max_abs": (env.terminal_joint_pos - before_joint).abs().max().item(),
                        "terminated": bool(terminated.item()),
                        "truncated": bool(truncated.item()),
                    }
                )
                if bool(terminated.item() or truncated.item()):
                    break
        data = {
            "checkpoint": str(args.checkpoint),
            "seed": args.seed,
            "observation_finite": bool(torch.isfinite(obs["policy"]).all()),
            "clip_actions": args.clip_actions,
            "half_range": args.half_range,
            "action_offset_min_max": [env.action_offset.min().item(), env.action_offset.max().item()],
            "action_scale_min_max": [env.action_scale.min().item(), env.action_scale.max().item()],
            "initial_amp_logit": initial_logit.view(-1).cpu().tolist(),
            "initial_style_reward": initial_style_reward.view(-1).cpu().tolist(),
            "steps_requested": args.steps,
            "steps_completed": len(steps),
            "steps": steps,
            "step_dt_s": env.step_dt,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + chr(10))
        print(json.dumps(data, allow_nan=False), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
