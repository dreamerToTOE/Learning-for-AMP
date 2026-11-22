"""Run PPO on the official Humanoid AMP Walk environment for R1 smoke testing.

The upstream v2.1.1 task registers only ``skrl_amp_cfg_entry_point``. This
wrapper injects a temporary PPO entry after Isaac Sim is launched, then
executes the unmodified upstream training script in memory.
"""

from __future__ import annotations

import os
from pathlib import Path


TASK_NAME = "Isaac-Humanoid-AMP-Walk-Direct-v0"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    isaaclab_root = os.environ.get("ISAACLAB_R1_ROOT")
    if not isaaclab_root:
        raise RuntimeError("ISAACLAB_R1_ROOT must point to the isolated Isaac Lab v2.1.1 checkout")

    ppo_cfg = project_root / "configs" / "r1" / "skrl_ppo_amp_smoke.yaml"
    train_script = Path(isaaclab_root) / "scripts" / "reinforcement_learning" / "skrl" / "train.py"
    if not ppo_cfg.is_file():
        raise FileNotFoundError(ppo_cfg)
    if not train_script.is_file():
        raise FileNotFoundError(train_script)

    source = train_script.read_text(encoding="utf-8")
    marker = "simulation_app = app_launcher.app"
    injection = f'''{marker}

# Register the temporary PPO config only after AppLauncher has initialized.
import gymnasium as gym
import isaaclab_tasks  # noqa: F401

gym.spec({TASK_NAME!r}).kwargs["skrl_cfg_entry_point"] = {str(ppo_cfg)!r}
'''
    if source.count(marker) != 1:
        raise RuntimeError(f"expected one AppLauncher marker in {train_script}")

    patched_source = source.replace(marker, injection, 1)
    globals_dict = {"__name__": "__main__", "__file__": str(train_script), "__package__": None}
    exec(compile(patched_source, str(train_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
