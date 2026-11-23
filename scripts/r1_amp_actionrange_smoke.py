"""Run the official AMP trainer with the opt-in R1 action-range correction."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    isaaclab_root = os.environ.get("ISAACLAB_R1_ROOT")
    if not isaaclab_root:
        raise RuntimeError("ISAACLAB_R1_ROOT must point to the isolated Isaac Lab v2.1.1 checkout")

    train_script = Path(isaaclab_root) / "scripts" / "reinforcement_learning" / "skrl" / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(train_script)

    source = train_script.read_text(encoding="utf-8")
    marker = "simulation_app = app_launcher.app"
    injection = f'''{marker}

# Apply the project-local action correction after Isaac Sim starts and before
# gym.make creates the Humanoid AMP environment.
from r1_action_range_patch import apply_action_range_patch
apply_action_range_patch()
'''
    if source.count(marker) != 1:
        raise RuntimeError(f"expected one AppLauncher marker in {train_script}")

    patched_source = source.replace(marker, injection, 1)
    globals_dict = {"__name__": "__main__", "__file__": str(train_script), "__package__": None}
    exec(compile(patched_source, str(train_script), "exec"), globals_dict)


if __name__ == "__main__":
    main()
