"""Run the fair speed-tracking PPO or AMP smoke without editing Isaac Lab."""

from __future__ import annotations

import os
from pathlib import Path


TASK_NAME = "Isaac-Humanoid-AMP-Walk-Direct-v0"
TARGET_VX_MPS = 1.5
SIGMA_VX_MPS = 0.5
PPO_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "r1" / "skrl_ppo_speed_tracking.yaml"


def main() -> None:
    isaaclab_root = os.environ.get("ISAACLAB_R1_ROOT")
    if not isaaclab_root:
        raise RuntimeError("ISAACLAB_R1_ROOT must point to the isolated Isaac Lab v2.1.1 checkout")
    train_script = Path(isaaclab_root) / "scripts" / "reinforcement_learning" / "skrl" / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    if not PPO_CONFIG.is_file():
        raise FileNotFoundError(PPO_CONFIG)

    source = train_script.read_text(encoding="utf-8")
    marker = "simulation_app = app_launcher.app"
    injection = f'''{marker}

# Install the fixed task reward after Isaac Sim starts and before the task is made.
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from pathlib import Path
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv

_R1_SPEED_TARGET_VX = {TARGET_VX_MPS!r}
_R1_SPEED_SIGMA_VX = {SIGMA_VX_MPS!r}
_r1_original_make = gym.make


def _r1_speed_tracking_rewards(self):
    vx = self.robot.data.body_lin_vel_w[:, self.ref_body_index, 0]
    height = self.robot.data.body_pos_w[:, self.ref_body_index, 2]
    alive = (height >= self.cfg.termination_height).to(vx.dtype)
    speed_error = vx - _R1_SPEED_TARGET_VX
    speed_bonus = torch.exp(-0.5 * (speed_error / _R1_SPEED_SIGMA_VX).square())
    self._r1_speed_metrics = {{
        "alive_reward": alive.detach(),
        "speed_bonus": speed_bonus.detach(),
        "task_reward": (alive * (1.0 + speed_bonus)).detach(),
        "torso_vx": vx.detach(),
        "speed_abs_error": speed_error.abs().detach(),
    }}
    return alive * (1.0 + speed_bonus)


HumanoidAmpEnv._get_rewards = _r1_speed_tracking_rewards


class _R1SpeedMetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._path = os.environ.get("R1_SPEED_METRICS_PATH")
        self._flush_steps = int(os.environ.get("R1_SPEED_METRICS_FLUSH_STEPS", "16"))
        self._step_count = 0
        self._sum = {{name: 0.0 for name in ("alive_reward", "speed_bonus", "task_reward", "torso_vx", "speed_abs_error")}}
        self._samples = 0
        if self._path:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            Path(self._path).write_text("", encoding="utf-8")

    def step(self, action):
        result = self.env.step(action)
        metrics = getattr(self.env.unwrapped, "_r1_speed_metrics", None)
        if metrics is not None:
            count = int(metrics["task_reward"].numel())
            self._samples += count
            self._step_count += 1
            for name, values in metrics.items():
                self._sum[name] += float(values.float().sum().item())
            if self._path and self._step_count >= self._flush_steps:
                self._flush()
        return result

    def _flush(self):
        if not self._path or not self._samples:
            return
        record = {{"environment_steps": self._step_count, "samples": self._samples}}
        record.update({{name: value / self._samples for name, value in self._sum.items()}})
        with Path(self._path).open("a", encoding="utf-8") as stream:
            import json
            stream.write(json.dumps(record, allow_nan=False) + "\\n")
        self._step_count = 0
        self._samples = 0
        self._sum = {{name: 0.0 for name in self._sum}}

    def close(self):
        self._flush()
        return self.env.close()


def _r1_make(*args, **kwargs):
    return _R1SpeedMetricsWrapper(_r1_original_make(*args, **kwargs))


gym.make = _r1_make
if args_cli.algorithm == "PPO":
    gym.spec({TASK_NAME!r}).kwargs["skrl_cfg_entry_point"] = {str(PPO_CONFIG)!r}
'''
    if source.count(marker) != 1:
        raise RuntimeError(f"expected one AppLauncher marker in {{train_script}}")
    patched_source = source.replace(marker, injection, 1)
    globals_dict = {"__name__": "__main__", "__file__": str(train_script), "__package__": None}
    exec(compile(patched_source, str(train_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
