"""Patch local Humanoid28 foot collision transforms before physics creation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


FOOT_PATHS = (
    "/pelvis/right_foot/collisions/right_foot",
    "/pelvis/left_foot/collisions/left_foot",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--offset-z", type=float, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from pxr import Gf, Usd

        stage = Usd.Stage.Open(str(args.mesh))
        if stage is None:
            raise RuntimeError(f"failed to open {args.mesh}")
        edits = []
        for path in FOOT_PATHS:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"missing mesh collision prim {path}")
            attr = prim.GetAttribute("xformOp:transform")
            matrix = Gf.Matrix4d(attr.Get())
            old_z = float(matrix[3][2])
            matrix.SetRow(3, Gf.Vec4d(matrix[3][0], matrix[3][1], old_z + args.offset_z, 1.0))
            attr.Set(matrix)
            edits.append({"path": path, "old_translation_z": old_z, "new_translation_z": float(matrix[3][2])})
        stage.GetRootLayer().Save()
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps({"mesh": str(args.mesh), "offset_z": args.offset_z, "edits": edits}, indent=2) + "\n")
        print(json.dumps({"mesh": str(args.mesh), "offset_z": args.offset_z, "edits": edits}), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
