"""Convert pinned HOLD G1 clips to the native MimicKit AMP motion format."""

import argparse
import hashlib
import json
import pickle
import urllib.request
from pathlib import Path

import h5py
import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def get_source(cache, revision, clip):
    path = cache / clip["path"]
    if path.is_file():
        content = path.read_bytes()
    else:
        url = f"https://media.githubusercontent.com/media/rai-opensource/hold/{revision}/{clip['path']}"
        with urllib.request.urlopen(url, timeout=60) as download:
            content = download.read()
    if sha256(content) != clip["sha256"]:
        raise ValueError(f"HOLD LFS digest mismatch for {clip['path']}")
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def get_foot_geoms(model, side):
    body = model.body(f"{side}_ankle_roll_link").id
    ids = np.flatnonzero((model.geom_bodyid == body) & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
    if ids.size == 0:
        raise ValueError(f"No sole sphere geometries found for {side} foot")
    return ids


def sole_heights(model, position, quaternion, joints):
    data = mujoco.MjData(model)
    feet = [get_foot_geoms(model, side) for side in ("left", "right")]
    heights = np.empty((len(joints), 2))
    for i, qpos in enumerate(np.column_stack((position, quaternion, joints))):
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        for side, ids in enumerate(feet):
            heights[i, side] = np.min(data.geom_xpos[ids, 2] - model.geom_size[ids, 0])
    return heights


def describe(samples):
    return {
        "mean": float(np.mean(samples)),
        "p10": float(np.quantile(samples, 0.10)),
        "median": float(np.median(samples)),
        "p90": float(np.quantile(samples, 0.90)),
    }


def convert(path, clip, model, joint_names, output):
    start, stop = clip["frames"]
    with h5py.File(path) as source:
        trajectory = source["robot_data/trajectory_data"]
        labels = trajectory["joints/labels"].asstr()[:].tolist()
        quat_order = source["robot_data/meta_data/quat_order"].asstr()[:].tolist()
        base_labels = trajectory["bodies/labels"].asstr()[:].tolist()
        if len(labels) != 29 or set(labels) != set(joint_names):
            raise ValueError(f"Unexpected joint names in {path}")
        if quat_order != list("wxyz") or base_labels != ["base"]:
            raise ValueError(f"Unexpected base or quaternion convention in {path}")
        if not 0 <= start < stop <= len(trajectory["time"]) or stop - start < 3:
            raise ValueError(f"Invalid crop [{start}, {stop}) for {path}")
        time = trajectory["time"][start:stop]
        position = trajectory["bodies/pos_w"][start:stop, 0]
        quaternion = trajectory["bodies/quat_w"][start:stop, 0]
        joints = trajectory["joints/pos"][start:stop][:, [labels.index(name) for name in joint_names]]
        reported_velocity = trajectory["bodies/lin_vel_w"][start:stop, 0]

    if not all(np.isfinite(value).all() for value in (time, position, quaternion, joints, reported_velocity)):
        raise ValueError(f"Non-finite poses or timestamps in {path}")
    dt = float(np.median(np.diff(time)))
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=1e-5, atol=1e-9):
        raise ValueError(f"Nonuniform sampled times in {path}")
    if np.max(np.abs(np.linalg.norm(quaternion, axis=1) - 1.0)) > 1e-4:
        raise ValueError(f"Non-unit source quaternion in {path}")
    lower, upper = model.jnt_range[1:].T
    if np.any((joints < lower - 1e-4) | (joints > upper + 1e-4)):
        raise ValueError(f"Source joint angle exceeds the G1 MJCF limits in {path}")

    position[:, :2] -= position[0, :2].copy()
    heading = np.arctan2(position[-1, 1], position[-1, 0])
    alignment = Rotation.from_rotvec([0.0, 0.0, -heading])
    position = alignment.apply(position)
    rotation = alignment * Rotation.from_quat(quaternion, scalar_first=True)
    feet = sole_heights(model, position, rotation.as_quat(scalar_first=True), joints)
    z_offset = float(clip.get("z_offset_m", 0.0))
    position[:, 2] += z_offset
    feet += z_offset

    frames = np.column_stack((position, rotation.as_rotvec(), joints)).astype(np.float32)
    fps = 1 / dt
    payload = pickle.dumps({"fps": fps, "loop_mode": 0, "frames": frames.tolist()}, protocol=4)
    motion = output / f"{clip['skill']}_{path.stem}.pkl"
    motion.write_bytes(payload)

    velocity = np.gradient(position, dt, axis=0)
    yaw = np.unwrap(rotation.as_euler("xyz")[:, 2])
    sine, cosine = np.sin(yaw), np.cos(yaw)
    forward = cosine * velocity[:, 0] + sine * velocity[:, 1]
    lateral = -sine * velocity[:, 0] + cosine * velocity[:, 1]
    supported = feet < 0.02
    exclusive = supported[supported.sum(axis=1) == 1, 0]
    return {
        **clip,
        "motion_file": motion.name,
        "motion_sha256": sha256(payload),
        "fps": fps,
        "n_frames": len(joints),
        "seconds": (len(joints) - 1) / fps,
        "loop": "clamp",
        "source_order_matches_mjcf": labels == joint_names,
        "alignment_yaw_deg": float(-np.rad2deg(heading)),
        "z_offset_m": z_offset,
        "forward_m_s": describe(forward),
        "lateral_m_s": describe(lateral),
        "yaw_rate_rad_s": describe(np.gradient(yaw, dt)),
        "hip_height_m": describe(position[:, 2]),
        "sole_lowest_m": describe(feet.min(axis=1)),
        "proxy_no_support_fraction": float(np.mean(~supported.any(axis=1))),
        "proxy_support_switches": int(np.count_nonzero(exclusive[1:] != exclusive[:-1])),
        "joint_positions_min_rad": joints.min(axis=0).tolist(),
        "joint_positions_max_rad": joints.max(axis=0).tolist(),
        "joint_velocity_max_abs_rad_s": float(np.max(np.abs(np.gradient(joints, dt, axis=0)))),
        "root_velocity_source_median_error_m_s": float(np.median(
            np.linalg.norm(velocity - alignment.apply(reported_velocity), axis=1)
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.manifest.read_text())
    upstream = args.upstream.resolve(strict=True)
    xml = upstream / "data/assets/g1/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected the floating-base 29 DoF G1 model")
    joint_names = [model.joint(index).name for index in range(1, model.njnt)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = [
        convert(get_source(args.cache, selection["commit"], clip), clip, model, joint_names, output)
        for clip in selection["clips"]
    ]
    dataset = output / "motions.yaml"
    dataset.write_text(yaml.safe_dump({
        "motions": [{"file": str(output / record["motion_file"]), "weight": 1.0} for record in records],
    }, sort_keys=False))
    env = yaml.safe_load((upstream / "data/envs/amp_g1_env.yaml").read_text())
    env["motion_file"] = str(dataset)
    (output / "amp_g1_walk_jog_env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
    agent = yaml.safe_load((upstream / "data/agents/amp_g1_agent.yaml").read_text())
    agent["iters_per_output"] = 250
    (output / "amp_g1_walk_jog_agent.yaml").write_text(yaml.safe_dump(agent, sort_keys=False))
    (output / "audit.json").write_text(json.dumps({
        **selection, "robot_xml_sha256": sha256(xml.read_bytes()), "joint_names": joint_names,
        "clip_weight": 1.0, "clips": records,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
