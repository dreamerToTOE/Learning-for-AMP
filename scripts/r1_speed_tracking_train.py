"""Run the project speed-tracking PPO or AMP task without editing Isaac Lab."""

from __future__ import annotations

import os
import sys
from pathlib import Path


TASK_NAME = "Isaac-Humanoid-AMP-Walk-Direct-v0"
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
    upright_task = "--upright-task" in sys.argv[1:]
    if upright_task:
        sys.argv.remove("--upright-task")
    marker = "simulation_app = app_launcher.app"
    injection = f'''{marker}

# Register the project-local task after Isaac Sim starts.
import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from pathlib import Path
from r1_speed_task import RULE_NAMES, SpeedTrackingWalkEnv, install_training_style_log

_r1_task_spec = gym.spec({TASK_NAME!r})
_r1_task_spec.entry_point = SpeedTrackingWalkEnv
_r1_task_spec.kwargs["upright_task"] = {upright_task!r}
if args_cli.algorithm == "PPO":
    _r1_task_spec.kwargs["skrl_cfg_entry_point"] = {str(PPO_CONFIG)!r}
_r1_original_make = gym.make


class _R1SpeedMetricsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._path = os.environ.get("R1_SPEED_METRICS_PATH")
        self._flush_steps = int(os.environ.get("R1_SPEED_METRICS_FLUSH_STEPS", "16"))
        self._step_count = 0
        self._sum = {{}}
        self._samples = 0
        if self._path:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            Path(self._path).write_text("", encoding="utf-8")

    def step(self, action):
        result = self.env.step(action)
        if self._path:
            speed_env = self.env.unwrapped
            metrics = dict(speed_env._r1_speed_metrics)
            if speed_env.upright_task:
                metrics["original_fall"] = speed_env.original_terminated
                for index, name in enumerate(RULE_NAMES):
                    metrics["terminate_" + name] = speed_env.rule_terminated[:, index]
            count = int(metrics["task_reward"].numel())
            self._samples += count
            self._step_count += 1
            totals = torch.stack(tuple(metrics.values())).sum(dim=1).tolist()
            for name, total in zip(metrics, totals):
                self._sum[name] = self._sum.get(name, 0.0) + total
            if self._step_count >= self._flush_steps:
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


def _r1_make(task, *args, **kwargs):
    env = _r1_original_make(task, *args, **kwargs)
    return _R1SpeedMetricsWrapper(env) if task == {TASK_NAME!r} else env


gym.make = _r1_make
'''
    if source.count(marker) != 1:
        raise RuntimeError(f"expected one AppLauncher marker in {train_script}")
    patched_source = source.replace(marker, injection, 1)
    runner_marker = "runner = Runner(env, agent_cfg)"
    if patched_source.count(runner_marker) != 1:
        raise RuntimeError(f"expected one skrl Runner marker in {train_script}")
    patched_source = patched_source.replace(
        runner_marker,
        runner_marker + '''
    if args_cli.algorithm == "AMP" and os.environ.get("R1_AMP_STYLE_METRICS_PATH"):
        install_training_style_log(runner.agent, Path(os.environ["R1_AMP_STYLE_METRICS_PATH"]))
''',
        1,
    )
    globals_dict = {"__name__": "__main__", "__file__": str(train_script), "__package__": None}
    exec(compile(patched_source, str(train_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
