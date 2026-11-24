"""Extract runtime Humanoid28 mass, COM, inertia and principal-axis data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        import torch

        task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
        cfg = parse_env_cfg(task, device=args.device, num_envs=1)
        env = gym.make(task, cfg=cfg)
        try:
            robot = env.unwrapped.robot
            view = robot.root_physx_view
            coms = view.get_coms()[0].detach().cpu()
            inertias = view.get_inertias()[0].detach().cpu()
            masses = robot.data.default_mass[0].detach().cpu()
            result = {
                "task": task,
                "device": str(args.device),
                "body_names": list(robot.data.body_names),
                "mass_kg": masses.tolist(),
                "coms_pose": coms.tolist(),
                "inertia_tensor": inertias.tolist(),
                "finite": bool(torch.isfinite(coms).all() and torch.isfinite(inertias).all()),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, allow_nan=False), flush=True)
        finally:
            env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
