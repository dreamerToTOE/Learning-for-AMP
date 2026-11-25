"""Render converted HOLD trajectories by setting G1 poses frame by frame."""

import argparse
import pickle
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def render(path, model, output):
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    frames = np.asarray(motion["frames"])
    annotations = motion.get("annotations")
    if annotations is not None and len(annotations) != len(frames):
        raise ValueError(f"Incorrect annotation count in {path}")
    poses = np.column_stack((
        frames[:, :3],
        Rotation.from_rotvec(frames[:, 3:6]).as_quat(scalar_first=True),
        frames[:, 6:],
    ))
    width, height, fps = 640, 360, 30
    sample_indices = np.rint(np.arange(0, (len(frames) - 1) / motion["fps"], 1 / fps) * motion["fps"]).astype(int)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth = 90
    camera.elevation = -12
    camera.distance = 2.65
    video = imageio_ffmpeg.write_frames(
        str(output), (width, height), fps=fps, codec="libx264", macro_block_size=1,
    )
    video.send(None)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for index in sample_indices:
                data.qpos[:] = poses[index]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = (*poses[index, :2], 0.70)
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                if annotations is not None:
                    labeled = Image.fromarray(frame)
                    draw = ImageDraw.Draw(labeled)
                    label = annotations[index]
                    right = draw.textbbox((12, 8), label)[2] + 6
                    draw.rectangle((6, 4, right, 24), fill="#131c28")
                    draw.text((12, 8), label, fill="white")
                    frame = np.asarray(labeled)
                video.send(frame)
    finally:
        video.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("motions", type=Path, nargs="+")
    args = parser.parse_args()
    spec = mujoco.MjSpec.from_file(str(args.xml.resolve(strict=True)))
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 360
    model = spec.compile()
    args.output.mkdir(parents=True, exist_ok=True)
    for motion in args.motions:
        render(motion, model, args.output / f"{motion.stem}.mp4")


if __name__ == "__main__":
    main()
