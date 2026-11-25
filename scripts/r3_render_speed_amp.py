"""Make annotated videos from recorded Newton policy poses."""

import argparse
import json
from pathlib import Path
import pickle
import tempfile

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from r3_preview_hold import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("cases", nargs="+")
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text())
    episodes = {episode["id"]: episode for episode in evaluation["episodes"]}
    names = evaluation["columns"]
    command_id = names.index("command_m_s")
    vx_id = names.index("vx_world")
    first_joint = names.index("qw") + 1
    last_joint = names.index("left_sole_force_n")
    dt = evaluation["dt"]
    spec = mujoco.MjSpec.from_file(str(args.xml.resolve(strict=True)))
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 360
    model = spec.compile()
    args.output.mkdir(parents=True, exist_ok=True)
    variant = evaluation["variant"]
    with np.load(args.evaluation.with_suffix(".npz")) as traces, tempfile.TemporaryDirectory() as tmp:
        motion = Path(tmp) / "motion.pkl"
        for case_id in args.cases:
            episode = episodes[case_id]
            trace = traces[case_id]
            assert len(trace) == episode["steps"]
            pose = np.column_stack((trace[:, :3], Rotation.from_quat(trace[:, 6:10]).as_rotvec(),
                                    trace[:, first_joint:last_joint]))
            frames = np.vstack((pose, pose[-1]))
            annotations = [
                f"{variant.capitalize()} D  |  {t * dt:.2f}s  |  command {command:.1f} m/s  |  vx {vx:+.2f} m/s"
                for t, (command, vx) in enumerate(trace[:, (command_id, vx_id)], start=1)
            ]
            if episode["terminal"] == "FAIL":
                annotations[-1] += "  |  fall"
            with motion.open("wb") as stream:
                pickle.dump({"fps": 1 / dt, "frames": frames, "annotations": [*annotations, annotations[-1]]}, stream)
            mp4 = args.output / f"{variant}_{case_id}.mp4"
            render(motion, model, mp4)
            print(mp4)


if __name__ == "__main__":
    main()
