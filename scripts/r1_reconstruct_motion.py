"""Reconstruct a phase-consistent 30 Hz motion file from the official 60 Hz clip."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    result[0] = (values[1] - values[0]) / dt
    result[-1] = (values[-1] - values[-2]) / dt
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = np.load(args.input, allow_pickle=False)
    keep = np.arange(0, source["dof_positions"].shape[0], 2)
    fps = float(source["fps"]) / 2.0
    dt = 1.0 / fps
    data = {
        "fps": np.asarray(fps, dtype=np.float32),
        "dof_names": source["dof_names"],
        "body_names": source["body_names"],
        "dof_positions": source["dof_positions"][keep].astype(np.float32),
        "dof_velocities": finite_difference(source["dof_positions"][keep], dt),
        "body_positions": source["body_positions"][keep].astype(np.float32),
        "body_rotations": source["body_rotations"][keep].astype(np.float32),
        "body_linear_velocities": finite_difference(source["body_positions"][keep], dt),
        "body_angular_velocities": source["body_angular_velocities"][keep].astype(np.float32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    print({"input_frames": int(len(source["dof_positions"])), "output_frames": int(len(keep)), "fps": fps})


if __name__ == "__main__":
    main()
