"""Create a reproducible Humanoid28 mass/inertia override layer for probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ASSET_URL = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
    "Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--asset-url", default=ASSET_URL)
    args = parser.parse_args()

    data = json.loads(args.audit.read_text())
    runtime = data["runtime"]
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "humanoid"',
        f"    subLayers = [@{args.asset_url}@]",
        ")",
        'over "humanoid" {',
    ]
    for name, mass, inertia in zip(
        runtime["body_names"], runtime["default_mass_kg"], runtime["default_inertia"]
    ):
        lines.extend(
            [
                f'    over "{name}" {{',
                f"        float physics:mass = {mass * args.scale:.9g}",
                "        float3 physics:diagonalInertia = "
                f"({inertia[0] * args.scale:.9g}, {inertia[4] * args.scale:.9g}, {inertia[8] * args.scale:.9g})",
                "    }",
            ]
        )
    lines.append("}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print({"output": str(args.output), "scale": args.scale, "body_count": len(runtime["body_names"])})


if __name__ == "__main__":
    main()
