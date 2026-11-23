"""Inspect local Humanoid28 layer references and prototype composition."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(args.asset)
        print({"root_layer": stage.GetRootLayer().identifier, "used_layers": [x.identifier for x in stage.GetUsedLayers()]})
        for path in (
            "/humanoid/right_foot/collisions/right_foot",
            "/humanoid/left_foot/collisions/left_foot",
        ):
            prim = stage.GetPrimAtPath(path)
            print(path, prim.IsValid(), prim.IsInstanceProxy(), prim.GetPrimInPrototype().GetPath() if prim.IsInstanceProxy() else None, prim.GetPrimStack())
    finally:
        app.close()


if __name__ == "__main__":
    main()
