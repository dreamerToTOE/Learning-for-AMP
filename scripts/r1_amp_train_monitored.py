"""Train the official Isaac Lab Humanoid AMP task with non-invasive AMP metrics."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="AMP", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--monitor_sample_size", type=int, default=512)
parser.add_argument("--monitor_interval", type=int, default=1, help="Record extra AMP metrics every N updates")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import skrl
from packaging import version

if version.parse(skrl.__version__) < version.parse("1.4.2"):
    raise RuntimeError(f"unsupported skrl version {skrl.__version__}")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm == "ppo" else f"skrl_{algorithm}_cfg_entry_point"


class TrainingMetricsWrapper(gym.Wrapper):
    """Collect episode counters without changing observations, rewards, or actions."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_steps = torch.zeros(env.unwrapped.num_envs, dtype=torch.int64, device=env.unwrapped.device)
        self._completed_lengths: list[int] = []
        self._falls = 0
        self._timeouts = 0

    def reset(self, **kwargs):
        self._episode_steps.zero_()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_steps += 1
        terminated = torch.as_tensor(terminated, device=self._episode_steps.device).view(-1).bool()
        truncated = torch.as_tensor(truncated, device=self._episode_steps.device).view(-1).bool()
        done = terminated | truncated
        if done.any():
            lengths = self._episode_steps[done].detach().cpu().tolist()
            self._completed_lengths.extend(int(value) for value in lengths)
            self._falls += int(terminated.sum().item())
            self._timeouts += int(truncated.sum().item())
            self._episode_steps[done] = 0
        return obs, reward, terminated, truncated, info

    def consume_episode_metrics(self):
        lengths = self._completed_lengths
        falls = self._falls
        timeouts = self._timeouts
        self._completed_lengths = []
        self._falls = 0
        self._timeouts = 0
        return lengths, falls, timeouts


def _restore_rng(states):
    torch.set_rng_state(states[0])
    if torch.cuda.is_available() and states[1] is not None:
        torch.cuda.set_rng_state(states[1])
    np.random.set_state(states[2])
    random.setstate(states[3])


def _save_rng():
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    return torch.get_rng_state(), cuda_state, np.random.get_state(), random.getstate()


def install_amp_monitor(agent, metric_env: TrainingMetricsWrapper, sample_size: int, monitor_interval: int, metric_path: Path):
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    original_update = agent._update
    previous_time = [time.perf_counter()]
    if monitor_interval < 1:
        raise ValueError("monitor_interval must be >= 1")

    def monitored_update(timestep, timesteps):
        update_number = int((timestep + 1) // agent.memory.memory_size)
        if monitor_interval > 1 and update_number != 1 and update_number % monitor_interval != 0:
            original_update(timestep, timesteps)
            return
        rng = _save_rng()
        try:
            memory = agent.memory
            rewards = memory.get_tensor_by_name("rewards")
            actions = memory.get_tensor_by_name("actions")
            states = memory.get_tensor_by_name("states")
            amp_states = memory.get_tensor_by_name("amp_states")
            flat_actions = actions.reshape(-1, actions.shape[-1])
            flat_states = states.reshape(-1, states.shape[-1])
            flat_amp_states = amp_states.reshape(-1, amp_states.shape[-1])
            monitor_count = min(sample_size, flat_amp_states.shape[0])
            if monitor_count < flat_amp_states.shape[0]:
                monitor_indices = torch.linspace(0, flat_amp_states.shape[0] - 1, monitor_count, device=flat_amp_states.device).long()
                monitor_amp_states = flat_amp_states[monitor_indices]
            else:
                monitor_amp_states = flat_amp_states
            with torch.no_grad():
                policy_input = agent._amp_state_preprocessor(monitor_amp_states)
                policy_logits, _, _ = agent.discriminator.act({"states": policy_input}, role="discriminator")
                policy_score = torch.sigmoid(policy_logits)
                policy_style_reward = -torch.log(torch.clamp(policy_score, min=1.0e-4)) * agent._discriminator_reward_scale
                expert_count = min(sample_size, 512)
                expert_states = metric_env.unwrapped.collect_reference_motions(expert_count)
                expert_input = agent._amp_state_preprocessor(expert_states)
                expert_logits, _, _ = agent.discriminator.act({"states": expert_input}, role="discriminator")
                expert_score = torch.sigmoid(expert_logits)

                raw_env = metric_env.unwrapped
                target = raw_env.action_offset + raw_env.action_scale * flat_actions
                limits = raw_env.robot.data.soft_joint_pos_limits[0]
                target_violations = ((target < limits[:, 0]) | (target > limits[:, 1])).float()
                joint_violation_ratio = target_violations.mean(dim=0)
                joint_names = list(raw_env.robot.data.joint_names)
                joint_positions = flat_states[:, : target.shape[-1]]
                target_error = (joint_positions - target).abs()

                lengths, falls, timeouts = metric_env.consume_episode_metrics()
                completed = len(lengths)
                elapsed = time.perf_counter() - previous_time[0]
                previous_time[0] = time.perf_counter()
                agent.track_data("Reward / Environment raw reward", rewards.mean().item())
                agent.track_data("Reward / AMP style reward", policy_style_reward.mean().item())
                agent.track_data("Reward / AMP weighted reward", (policy_style_reward * agent._style_reward_weight).mean().item())
                agent.track_data("Discriminator / Policy score", policy_score.mean().item())
                agent.track_data("Discriminator / Expert score", expert_score.mean().item())
                agent.track_data("Training / Samples per update", float(rewards.numel()))
                agent.track_data("Policy / Action min", flat_actions.min().item())
                agent.track_data("Policy / Action max", flat_actions.max().item())
                agent.track_data("Policy / Action absolute max", flat_actions.abs().max().item())
                agent.track_data("Joint / Target angle violation ratio", target_violations.mean().item())
                agent.track_data("Joint / Target position error mean", target_error.mean().item())
                agent.track_data("Joint / Target position error max", target_error.max().item())
                agent.track_data("Training / Environment steps per second", rewards.numel() / max(elapsed, 1.0e-6))
                agent.track_data("Episode / Completed episodes", float(completed))
                agent.track_data("Episode / Fall rate", falls / completed if completed else 0.0)
                agent.track_data("Episode / Timeout rate", timeouts / completed if completed else 0.0)
                average_survival_steps = float(sum(lengths) / completed) if completed else None
                average_survival_seconds = (
                    average_survival_steps * metric_env.unwrapped.step_dt if average_survival_steps is not None else None
                )
                if completed:
                    agent.track_data("Episode / Average survival steps", average_survival_steps)
                    agent.track_data("Episode / Average survival seconds", average_survival_seconds)
                record = {
                    "timestep": int(timestep),
                    "trainer_timesteps": int(timesteps),
                    "samples_per_update": int(rewards.numel()),
                    "monitor_amp_score_samples": int(monitor_count),
                    "environment_reward_mean": float(rewards.mean().item()),
                    "amp_style_reward_mean": float(policy_style_reward.mean().item()),
                    "amp_weighted_reward_mean": float((policy_style_reward * agent._style_reward_weight).mean().item()),
                    "policy_discriminator_score_mean": float(policy_score.mean().item()),
                    "expert_discriminator_score_mean": float(expert_score.mean().item()),
                    "action_min": float(flat_actions.min().item()),
                    "action_max": float(flat_actions.max().item()),
                    "action_absolute_max": float(flat_actions.abs().max().item()),
                    "target_angle_violation_ratio": float(target_violations.mean().item()),
                    "target_angle_violation_by_joint": {
                        name: float(value) for name, value in zip(joint_names, joint_violation_ratio.detach().cpu().tolist())
                    },
                    "target_position_error_mean": float(target_error.mean().item()),
                    "target_position_error_max": float(target_error.max().item()),
                    "completed_episodes": completed,
                    "fall_count": int(falls),
                    "timeout_count": int(timeouts),
                    "fall_rate": float(falls / completed) if completed else 0.0,
                    "average_survival_steps": average_survival_steps,
                    "average_survival_seconds": average_survival_seconds,
                    "environment_steps_per_second": float(rewards.numel() / max(elapsed, 1.0e-6)),
                }
                with metric_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
        finally:
            _restore_rng(rng)
        original_update(timestep, timesteps)

    agent._update = monitored_update


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line: {log_dir_name}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir_name += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir_name
    log_dir = os.path.join(log_root_path, log_dir_name)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm == "ppo":
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    metric_env = TrainingMetricsWrapper(env)
    env = SkrlVecEnvWrapper(metric_env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)
    install_amp_monitor(
        runner.agent, metric_env, args_cli.monitor_sample_size, args_cli.monitor_interval, Path(log_dir) / "amp_metrics.jsonl"
    )
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
