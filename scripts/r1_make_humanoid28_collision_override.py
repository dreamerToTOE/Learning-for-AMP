"""Create a local flattened Humanoid28 USD with a controlled foot-collision offset.

The shipped Humanoid28 asset composes the collision cubes through instanceable
mesh payloads.  Runtime authoring on those instance proxies is rejected by USD
and can also leave PhysX with a different shape than the authored stage.  This
utility resolves the asset first, edits the flattened prototype layer, and
exports a self-contained USD that can be selected through the robot spawn
configuration before physics initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


ASSET_URL = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
    "Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd"
)
FOOT_COLLISION_PATHS = (
    "/humanoid/right_foot/collisions/right_foot",
    "/humanoid/left_foot/collisions/left_foot",
)


def patch_asset(Usd, Gf, input_url: str, output: Path, offset_z: float) -> dict:
    source_stage = Usd.Stage.Open(input_url)
    if source_stage is None:
        raise RuntimeError(f"failed to open Humanoid28 asset: {input_url}")

    # Flattening resolves the instanceable mesh payload. Export first because
    # Isaac Sim 4.5 cannot reliably open an in-memory flattened Sdf layer.
    flattened_layer = source_stage.Flatten()
    output.parent.mkdir(parents=True, exist_ok=True)
    flattened_layer.Export(str(output))
    flattened_stage = Usd.Stage.Open(str(output))
    if flattened_stage is None:
        raise RuntimeError("failed to open flattened Humanoid28 stage")

    edits = []
    for instance_path in FOOT_COLLISION_PATHS:
        instance_prim = flattened_stage.GetPrimAtPath(instance_path)
        if not instance_prim.IsValid() or not instance_prim.IsInstanceProxy():
            raise RuntimeError(f"expected an instance proxy at {instance_path}")
        prototype_path = str(instance_prim.GetPrimInPrototype().GetPath())

        # Author a local override on the instance path. Editing the prototype
        # directly is rejected by USD, even after flattening.
        container = instance_prim.GetParent()
        container.SetInstanceable(False)
        instance_prim = flattened_stage.GetPrimAtPath(instance_path)
        attribute = instance_prim.GetAttribute("xformOp:transform")
        matrix = Gf.Matrix4d(attribute.Get())
        translation = matrix.ExtractTranslation()
        old_z = float(translation[2])
        # SetTranslate resets the 3x3 scale in this USD build. Updating only
        # the translation row preserves the collision cube dimensions.
        matrix.SetRow(
            3,
            Gf.Vec4d(translation[0], translation[1], translation[2] + offset_z, 1.0),
        )
        attribute.Set(matrix)
        edits.append(
            {
                "instance_path": instance_path,
                "prototype_path": prototype_path,
                "attribute_path": f"{instance_path}.xformOp:transform",
                "old_translation_z": old_z,
                "new_translation_z": float(matrix.ExtractTranslation()[2]),
            }
        )

    flattened_stage.GetRootLayer().Save()
    return {
        "input_asset": input_url,
        "output_asset": str(output),
        "offset_z": offset_z,
        "edits": edits,
        "flattened_root_prims": [str(prim.path) for prim in flattened_layer.rootPrims],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset-z", type=float, default=0.05)
    parser.add_argument("--input-url", default=ASSET_URL)
    parser.add_argument("--audit-output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from pxr import Gf, Usd

        result = patch_asset(Usd, Gf, args.input_url, args.output, args.offset_z)
        if args.audit_output:
            args.audit_output.parent.mkdir(parents=True, exist_ok=True)
            args.audit_output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
