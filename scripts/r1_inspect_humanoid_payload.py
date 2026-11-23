"""Inspect the Humanoid28 USD and mesh payload layers directly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


ASSET_URL = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd"
MESH_URL = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28/Props/instanceable_meshes.usd"


def inspect_stage(Usd, UsdGeom, UsdPhysics, url: str) -> dict:
    stage = Usd.Stage.Open(url)
    prims = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = path.lower()
        if not any(token in name for token in ("foot", "collision", "torso", "pelvis", "knee", "elbow")):
            continue
        boundable = UsdGeom.Boundable(prim)
        prims.append(
            {
                "path": path,
                "type": prim.GetTypeName(),
                "rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                "collision": prim.HasAPI(UsdPhysics.CollisionAPI),
                "extent": boundable.GetExtentAttr().Get() if boundable else None,
                "attributes": {
                    attr.GetName(): attr.Get()
                    for attr in prim.GetAttributes()
                    if any(token in attr.GetName().lower() for token in ("mass", "inertia", "collision", "material", "rest", "friction", "xformop"))
                },
            }
        )
    root_spec = stage.GetRootLayer().GetPrimAtPath("/humanoid")
    root_prim = stage.GetPrimAtPath("/humanoid")
    return {
        "url": url,
        "root_layer": str(stage.GetRootLayer().identifier),
        "used_layers": [str(layer.identifier) for layer in stage.GetUsedLayers()],
        "prims": prims,
        "root_prim_references": [str(item) for item in root_spec.referenceList.GetAddedOrExplicitItems()] if root_spec else [],
        "root_prim_payloads": [str(item) for item in root_spec.payloadList.GetAddedOrExplicitItems()] if root_spec else [],
        "composed_root_references": [str(item) for item in root_prim.GetReferences().GetAddedOrExplicitItems()] if root_prim.IsValid() else [],
        "composed_root_payloads": [str(item) for item in root_prim.GetPayloads().GetAddedOrExplicitItems()] if root_prim.IsValid() else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from pxr import Usd, UsdGeom, UsdPhysics

        result = {
            "asset": inspect_stage(Usd, UsdGeom, UsdPhysics, ASSET_URL),
            "mesh": inspect_stage(Usd, UsdGeom, UsdPhysics, MESH_URL),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps(result, default=str), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
