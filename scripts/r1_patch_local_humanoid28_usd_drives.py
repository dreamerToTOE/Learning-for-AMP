"""Patch revolute joint drives in a local Humanoid28 root USD layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


JOINTS = {
    "right_elbow": "X",
    "left_elbow": "X",
    "right_knee": "X",
    "left_knee": "X",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--stiffness", type=float, required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument("--max-force", type=float, default=1.0e9)
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(args.asset))
        if stage is None:
            raise RuntimeError(f"failed to open {args.asset}")
        edits = []
        for joint, axis in JOINTS.items():
            prim = stage.GetPrimAtPath(f"/humanoid/joints/{joint}")
            if not prim.IsValid():
                raise RuntimeError(f"missing joint {joint}")
            values = {
                f"drive:{axis}:physics:stiffness": args.stiffness,
                f"drive:{axis}:physics:damping": args.damping,
                f"drive:{axis}:physics:maxForce": args.max_force,
            }
            for name, value in values.items():
                attr = prim.GetAttribute(name)
                if not attr.IsValid():
                    raise RuntimeError(f"missing attribute {name} on {joint}")
                old = attr.Get()
                attr.Set(value)
                edits.append({"joint": joint, "attribute": name, "old": old, "new": value})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stage.GetRootLayer().Export(str(args.output))
        result = {"input": str(args.asset), "output": str(args.output), "stiffness": args.stiffness, "damping": args.damping, "max_force": args.max_force, "edits": edits}
        print(json.dumps(result, indent=2), flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
