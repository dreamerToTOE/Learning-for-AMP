"""Inspect the loaded Humanoid28 USD joint drives through Isaac Sim."""

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
        inspect_usd(args)
    finally:
        app.close()


def inspect_usd(args) -> None:
    import gymnasium as gym
    import omni.usd
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from pxr import UsdPhysics

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    try:
        stage = omni.usd.get_context().get_stage()
        joints = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/World/envs/env_0/Robot/joints/" not in path:
                continue
            attributes = {}
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if any(token in name.lower() for token in ("stiff", "damping", "maxforce", "drive")):
                    attributes[name] = attr.Get()
            joint = {
                "path": path,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "attributes": attributes,
            }
            for drive_type in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
                if drive:
                    joint[f"{drive_type}_drive"] = {
                        "stiffness": drive.GetStiffnessAttr().Get(),
                        "damping": drive.GetDampingAttr().Get(),
                        "max_force": drive.GetMaxForceAttr().Get(),
                    }
            joints.append(joint)
        result = {
            "task": task,
            "asset_usd_path": env.unwrapped.robot.cfg.spawn.usd_path,
            "joint_count": len(joints),
            "joints": joints,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
