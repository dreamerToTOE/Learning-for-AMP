"""Classify unconditioned G1 policy rollouts using speed, posture and foot contacts."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def describe_windows(trace, dt, joint_names, protocol):
    root = Rotation.from_quat(trace[:, 6:10])
    matrix = root.as_matrix()
    yaw = np.unwrap(np.arctan2(matrix[:, 1, 0], matrix[:, 0, 0]))
    sine, cosine = np.sin(yaw), np.cos(yaw)
    vx = cosine * trace[:, 3] + sine * trace[:, 4]
    vy = -sine * trace[:, 3] + cosine * trace[:, 4]
    tilt = np.rad2deg(np.arccos(np.clip(matrix[:, 2, 2], -1, 1)))
    contacts = trace[:, -5:-3] >= protocol["contact_force_n"]
    window = round(protocol["window_s"] / dt)
    burn_in = round(protocol["burn_in_s"] / dt)
    knee_ids = [10 + joint_names.index(f"{side}_knee_joint") for side in ("left", "right")]

    results = []
    for start in range(burn_in, len(trace) - window + 1, window):
        interval = slice(start, start + window)
        feet = contacts[interval]
        exclusive = feet[feet.sum(axis=1) == 1, 0]
        switches = int(np.count_nonzero(exclusive[1:] != exclusive[:-1]))
        no_support = float(np.mean(~feet.any(axis=1)))
        forward = float(np.median(vx[interval]))
        lateral = float(np.mean(vy[interval]))
        yaw_rate = float((yaw[start + window - 1] - yaw[start]) / ((window - 1) * dt))
        height_p10 = float(np.quantile(trace[interval, 2], 0.10))
        tilt_p90 = float(np.quantile(tilt[interval], 0.90))
        nonfoot_force = float(trace[interval, -3].max())
        mode = "unclassified"
        if (switches >= protocol["min_support_switches"]
                and height_p10 >= protocol["min_pelvis_p10_m"]
                and tilt_p90 <= protocol["max_pelvis_tilt_p90_deg"]
                and abs(lateral) <= protocol["max_abs_mean_lateral_m_s"]
                and abs(yaw_rate) <= protocol["max_abs_mean_yaw_rate_rad_s"]
                and nonfoot_force < protocol["contact_force_n"]):
            for skill in ("walk", "jog"):
                low, high = protocol[skill]["forward_speed_m_s"]
                support_lo, support_hi = protocol[skill]["no_foot_support_fraction"]
                if low <= forward <= high and support_lo <= no_support <= support_hi:
                    mode = skill
                    break
        results.append({
            "start_s": start * dt, "mode": mode, "median_forward_m_s": forward,
            "mean_lateral_m_s": lateral, "mean_yaw_rate_rad_s": yaw_rate,
            "pelvis_height_p10_m": height_p10, "pelvis_tilt_p90_deg": tilt_p90,
            "foot_support_switches": switches, "no_foot_support_fraction": no_support,
            "nonfoot_peak_force_n": nonfoot_force,
            "knee_range_rad": np.ptp(trace[interval][:, knee_ids], axis=0).tolist(),
        })
    return results


def evaluate_episode(env, agent, case, joint_names, protocol, device):
    from envs.base_env import DoneFlags

    skill, seed, motion_id = case
    phase = float(np.random.default_rng(seed).uniform())
    ids = torch.tensor([motion_id], device=device)
    times = env._motion_lib.get_motion_length(ids) * phase
    original_sampler = env._sample_motion_times
    env._sample_motion_times = lambda n: (ids, times)
    try:
        obs, info = env.reset()
    finally:
        env._sample_motion_times = original_sampler
    initial_sha256 = hashlib.sha256(obs[0].cpu().numpy().tobytes()).hexdigest()

    engine = env._engine
    robot = env._get_char_id()
    names = engine.get_obj_body_names(robot)
    feet = [names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    nonfeet = [index for index in range(len(names)) if index not in feet]
    initial_pos = engine.get_root_pos(robot)[0].cpu().numpy().copy()
    trace = []
    for _ in range(protocol["max_steps"]):
        action, _ = agent._decide_action(obs, info)
        obs, _, done, info = env.step(action)
        force = engine.get_ground_contact_forces(robot)[0].norm(dim=-1)
        clipped = (action < env._action_bound_low) | (action > env._action_bound_high)
        row = torch.cat((
            engine.get_root_pos(robot)[0], engine.get_root_vel(robot)[0],
            engine.get_root_rot(robot)[0], engine.get_dof_pos(robot)[0], force[feet],
            force[nonfeet].max().reshape(1), clipped.float().mean().reshape(1),
            done.float(), (torch.isfinite(action).all() & torch.isfinite(obs).all()).float().reshape(1),
        )).cpu().numpy()
        if not row[-1] or not np.isfinite(row).all():
            raise ValueError(f"Non-finite simulator or policy state at seed {seed}")
        trace.append(row[:-1])
        terminal = DoneFlags(int(row[-2]))
        if terminal != DoneFlags.NULL:
            break

    trace = np.asarray(trace)
    dt = engine.get_timestep()
    windows = describe_windows(trace, dt, joint_names, protocol)
    labels = [window["mode"] for window in windows]
    streak = protocol["sustained_windows"]
    sustained = [
        mode for mode in ("walk", "jog")
        if terminal == DoneFlags.TIME
        and any(labels[i:i + streak] == [mode] * streak for i in range(len(labels) - streak + 1))
    ]
    displacement = trace[-1, :3] - initial_pos
    return {
        "initial_skill": skill, "motion_id": motion_id, "seed": seed, "initial_phase": phase,
        "initial_observation_sha256": initial_sha256, "terminal": terminal.name,
        "steps": len(trace), "duration_s": len(trace) * dt,
        "world_forward_m": float(displacement[0]), "world_lateral_m": float(displacement[1]),
        "target_clip_fraction": float(trace[:, -2].mean()),
        "sustained_skills": sustained, "windows": windows,
    }, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("fixed", "held_out"), default="fixed")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-seed", type=int)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()

    data = args.data.resolve(strict=True)
    upstream = args.upstream.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    video_path = args.video.resolve() if args.video else None
    audit = json.loads((data / "audit.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    cases = []
    for group, skill in enumerate(("walk", "jog")):
        clip_ids = [i for i, clip in enumerate(audit["clips"]) if clip["skill"] == skill]
        if not clip_ids:
            raise ValueError(f"No {skill} clip for the reset distribution")
        base = protocol[f"{args.split}_evaluation_seed"] + 100 * group
        for index in range(protocol[f"{args.split}_episodes_per_initial_skill"]):
            cases.append((skill, base + index, clip_ids[index % len(clip_ids)]))
    if args.case_seed is not None:
        cases = [case for case in cases if case[1] == args.case_seed]
    if not cases or (video_path and len(cases) != 1):
        parser.error("Select a valid --case-seed in the split when saving a video")

    os.chdir(upstream)
    sys.path.insert(0, str(upstream / "mimickit"))
    from envs.env_builder import build_env
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import mp_util

    mp_util.init(0, 1, args.device, None)
    env = build_env(
        str(data / "amp_g1_walk_jog_env.yaml"), "data/engines/newton_engine.yaml",
        1, args.device, visualize=False, record_video=video_path is not None,
    )
    agent = build_agent(str(data / "amp_g1_walk_jog_agent.yaml"), env, args.device)
    agent.load(str(checkpoint))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    episodes, traces = [], {}
    with torch.no_grad():
        for case in cases:
            episode, trace = evaluate_episode(env, agent, case, audit["joint_names"], protocol, args.device)
            episodes.append(episode)
            traces[f"seed_{episode['seed']}"] = trace
            print(f"{episode['seed']}: {episode['terminal']} {episode['duration_s']:.2f}s "
                  f"{episode['sustained_skills']}", flush=True)
    if video_path:
        recording = env.record_diagnostics()["sim_recording"]
        if recording.get_num_frames() != episodes[0]["steps"] or not any(
            np.any(frame) for frame in recording.get_frames()
        ):
            raise RuntimeError("The Newton recorder did not capture a visible episode")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        recording.save(str(video_path))

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), **traces)
    columns = (
        ["x", "y", "z", "vx_world", "vy_world", "vz_world", "qx", "qy", "qz", "qw"]
        + audit["joint_names"]
        + ["left_sole_force_n", "right_sole_force_n", "nonfoot_max_force_n", "clipped_fraction", "done"]
    )
    output.write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "split": args.split, "protocol": protocol, "trace_columns": columns, "episodes": episodes,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
