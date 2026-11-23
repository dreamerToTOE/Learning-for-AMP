"""Export the remote Humanoid28 root and mesh layers to local USD files."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


URLS = {
    "humanoid_28.usd": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd",
    "instanceable_meshes.usd": "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/IsaacLab/Robots/Classic/Humanoid28/Props/instanceable_meshes.usd",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from pxr import Usd

        args.output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for filename, url in URLS.items():
            stage = Usd.Stage.Open(url)
            if stage is None:
                raise RuntimeError(f"failed to open {url}")
            output = args.output_dir / filename
            stage.GetRootLayer().Export(str(output))
            results.append({"url": url, "output": str(output), "size": output.stat().st_size})
        print(results, flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
