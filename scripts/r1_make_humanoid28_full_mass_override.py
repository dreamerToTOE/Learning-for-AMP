"""Create a full Humanoid28 mass-property override from runtime PhysX data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ASSET_URL = (
    "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/"
    "Isaac/IsaacLab/Robots/Classic/Humanoid28/humanoid_28.usd"
)


def rotation_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        xyzw = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                scale / 4.0,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1.0e-12)) * 2.0
            xyzw = np.array(
                [
                    scale / 4.0,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1.0e-12)) * 2.0
            xyzw = np.array(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    scale / 4.0,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1.0e-12)) * 2.0
            xyzw = np.array(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    scale / 4.0,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    xyzw /= np.linalg.norm(xyzw)
    return float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--properties", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-url", default=ASSET_URL)
    args = parser.parse_args()

    data = json.loads(args.properties.read_text())
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "humanoid"',
        f"    subLayers = [@{args.asset_url}@]",
        ")",
        'over "humanoid" {',
    ]
    for name, mass, com, inertia in zip(
        data["body_names"], data["mass_kg"], data["coms_pose"], data["inertia_tensor"]
    ):
        tensor = np.asarray(inertia, dtype=np.float64).reshape(3, 3)
        eigenvalues, eigenvectors = np.linalg.eigh(tensor)
        if np.linalg.det(eigenvectors) < 0.0:
            eigenvectors[:, -1] *= -1.0
        qw, qx, qy, qz = rotation_to_quat_wxyz(eigenvectors)
        lines.extend(
            [
                f'    over "{name}" {{',
                f"        float physics:mass = {float(mass):.9g}",
                f"        point3f physics:centerOfMass = ({float(com[0]):.9g}, {float(com[1]):.9g}, {float(com[2]):.9g})",
                f"        float3 physics:diagonalInertia = ({eigenvalues[0]:.9g}, {eigenvalues[1]:.9g}, {eigenvalues[2]:.9g})",
                f"        quatf physics:principalAxes = ({qw:.9g}, {qx:.9g}, {qy:.9g}, {qz:.9g})",
                "    }",
            ]
        )
    lines.append("}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print({"output": str(args.output), "body_count": len(data["body_names"])})


if __name__ == "__main__":
    main()
