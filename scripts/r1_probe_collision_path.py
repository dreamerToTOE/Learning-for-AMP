"""Check the composed foot collision prim paths without editing them."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        import json
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import omni.usd
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
        env = gym.make(task, cfg=parse_env_cfg(task, device=args.device, num_envs=1))
        stage = omni.usd.get_context().get_stage()
        result = []
        for path in (
            "/World/envs/env_0/Robot/right_foot/collisions/right_foot",
            "/World/envs/env_0/Robot/left_foot/collisions/left_foot",
        ):
            prim = stage.GetPrimAtPath(path)
            result.append(
                {
                    "path": path,
                    "valid": prim.IsValid(),
                    "active": prim.IsActive(),
                    "instance_proxy": prim.IsInstanceProxy(),
                    "prim_in_prototype": str(prim.GetPrimInPrototype().GetPath()) if prim.IsInstanceProxy() else None,
                    "prototype": str(prim.GetPrimInPrototype().GetStage().GetRootLayer().identifier)
                    if prim.IsInstanceProxy()
                    else None,
                    "type": prim.GetTypeName(),
                    "attrs": [attr.GetName() for attr in prim.GetAttributes()],
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        env.close()
    finally:
        app.close()


if __name__ == "__main__":
    main()
