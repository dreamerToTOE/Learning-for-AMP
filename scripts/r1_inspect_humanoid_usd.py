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
    from pxr import UsdGeom, UsdPhysics

    task = "Isaac-Humanoid-AMP-Walk-Direct-v0"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    try:
        stage = omni.usd.get_context().get_stage()
        joints = []
        foot_prims = []
        bodies = []
        collision_prims = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if "/World/envs/env_0/Robot/" in path and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(
                    {
                        "path": path,
                        "name": prim.GetName(),
                        "type": prim.GetTypeName(),
                        "attributes": {
                            attr.GetName(): attr.Get()
                            for attr in prim.GetAttributes()
                            if any(token in attr.GetName().lower() for token in ("mass", "inertia", "centerofmass", "principalaxes"))
                        },
                    }
                )
            if "/World/envs/env_0/Robot/" in path and prim.HasAPI(UsdPhysics.CollisionAPI):
                extent = None
                boundable = UsdGeom.Boundable(prim)
                if boundable:
                    extent = boundable.GetExtentAttr().Get()
                collision_prims.append({"path": path, "type": prim.GetTypeName(), "extent": extent})
            if "/World/envs/env_0/Robot/" in path and "foot" in path.lower():
                foot_prims.append(
                    {
                        "path": path,
                        "type": prim.GetTypeName(),
                        "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                    }
                )
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
            for attr_name in ("physics:axis", "physics:lowerLimit", "physics:upperLimit"):
                value = prim.GetAttribute(attr_name).Get()
                if value is not None:
                    joint[attr_name] = value
            for rel_name in ("physics:body0", "physics:body1"):
                targets = prim.GetRelationship(rel_name).GetTargets()
                if targets:
                    joint[rel_name] = [str(target) for target in targets]
            for drive_type in ("angular", "linear"):
                drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
                if drive:
                    joint[f"{drive_type}_drive"] = {
                        "stiffness": drive.GetStiffnessAttr().Get(),
                        "damping": drive.GetDampingAttr().Get(),
                        "max_force": drive.GetMaxForceAttr().Get(),
                    }
            joints.append(joint)
        robot = env.unwrapped.robot
        result = {
            "task": task,
            "asset_usd_path": env.unwrapped.robot.cfg.spawn.usd_path,
            "joint_count": len(joints),
            "body_count": len(bodies),
            "bodies": bodies,
            "foot_prims": foot_prims,
            "collision_prims": collision_prims,
            "joints": joints,
            "runtime": {
                "body_names": list(robot.data.body_names),
                "joint_names": list(robot.data.joint_names),
                "default_mass_kg": robot.data.default_mass[0].cpu().tolist(),
                "default_inertia": robot.data.default_inertia[0].cpu().tolist(),
                "soft_joint_pos_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
                "joint_vel_limits": robot.data.joint_vel_limits[0].cpu().tolist(),
                "joint_effort_limits": robot.data.joint_effort_limits[0].cpu().tolist(),
                "articulation_props": {
                    "solver_position_iteration_count": env.unwrapped.robot.cfg.spawn.articulation_props.solver_position_iteration_count,
                    "solver_velocity_iteration_count": env.unwrapped.robot.cfg.spawn.articulation_props.solver_velocity_iteration_count,
                    "enabled_self_collisions": env.unwrapped.robot.cfg.spawn.articulation_props.enabled_self_collisions,
                },
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
