"""Replay matched G1 initial states under fixed and switching speed commands."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import wasserstein_distance
import torch

from envs.base_env import DoneFlags
from learning.base_agent import AgentMode
from util import mp_util, torch_util, util

from r3_eval_amp_modes import describe_windows
from r3_speed_amp import FEET, VARIANTS, build_env_agent, prepare_configs


SCHEDULES = {"walk": ("walk", "walk"), "jog": ("jog", "jog"),
             "walk_to_jog": ("walk", "jog"), "jog_to_walk": ("jog", "walk")}


def build_cases(env, audit, protocol, split):
    setup = protocol["evaluation"]
    base = setup[f"{split}_seed"]
    count = setup[f"{split}_phases_per_initial_skill"]
    commands = protocol["commands_m_s"]
    by_file = {clip["motion_file"]: clip["skill"] for clip in audit["clips"]}
    motion_ids = {skill: [] for skill in commands}
    for motion_id in range(env._motion_lib.get_num_motions()):
        name = Path(env._motion_lib.get_motion_file(motion_id)).name
        motion_ids[by_file[name]].append(motion_id)

    cases = []
    for group, (skill, motions) in enumerate(motion_ids.items()):
        assert motions and count % len(motions) == 0
        for i in range(count):
            seed = base + 100 * group + i
            phase = float(np.random.default_rng(seed).uniform())
            for schedule, (first, last) in SCHEDULES.items():
                cases.append({
                    "id": f"{seed}_{schedule}", "seed": seed, "phase": phase,
                    "initial_skill": skill, "motion_id": motions[i % len(motions)],
                    "schedule": schedule, "initial_command_m_s": commands[first],
                    "final_command_m_s": commands[last],
                })
    assert len(cases) == env.get_num_envs()
    return cases


def sole_geometry(char_file, device):
    model = mujoco.MjModel.from_xml_path(char_file)
    ids = [
        np.flatnonzero((model.geom_bodyid == model.body(name).id)
                       & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
                       & (model.geom_contype != 0))
        for name in FEET
    ]
    assert len(ids[0]) == len(ids[1]) > 0
    centers = torch.tensor(model.geom_pos[ids], device=device, dtype=torch.float)
    radii = torch.tensor(model.geom_size[ids, 0], device=device, dtype=torch.float)
    return centers, radii


@torch.no_grad()
def roll_out(env, agent, cases, steps, switch_step, spheres):
    device = env._device
    phase = torch.tensor([case["phase"] for case in cases], device=device)
    ids = torch.tensor([case["motion_id"] for case in cases], device=device)
    times = phase * env._motion_lib.get_motion_length(ids)
    commands = [
        torch.tensor([case[f"{stage}_command_m_s"] for case in cases], device=device)
        for stage in ("initial", "final")
    ]
    env.auto_commands = False
    sampler = env._sample_motion_times
    try:
        env._sample_motion_times = lambda n: (ids, times)
        obs, info = env.reset()
    finally:
        env._sample_motion_times = sampler
    env.set_eval_speed(commands[0])
    physical_obs = obs[:, :-1].cpu().numpy()
    starts = {}
    for index, case in enumerate(cases):
        digest = hashlib.sha256(physical_obs[index].tobytes()).hexdigest()
        assert starts.setdefault(case["seed"], digest) == digest, "Command changed physical reset"
        case["initial_observation_sha256"] = digest
    assert len(starts) == len(set(starts.values()))

    engine, robot = env._engine, env._get_char_id()
    origins = engine.get_root_pos(robot).cpu().numpy().copy()
    centers, radii = spheres
    full_trace = []
    for step in range(steps):
        if step == switch_step:
            env.set_eval_speed(commands[1])
        action, _ = agent._decide_action(obs, info)
        obs, reward, done, info = env.step(action)
        body_pos = engine.get_body_pos(robot)[:, env.foot_ids]
        body_rot = engine.get_body_rot(robot)
        sole_offset = torch_util.quat_rotate(body_rot[:, env.foot_ids, None], centers[None])
        sole_z = body_pos[..., 2] + (sole_offset[..., 2] - radii).amin(dim=-1)
        contact = engine.get_ground_contact_forces(robot).norm(dim=-1)
        nonfoot_force = contact[:, env._nonfoot_ids].amax(dim=-1)
        clips = (action < env._action_bound_low) | (action > env._action_bound_high)
        torso_up = 1 - 2 * body_rot[:, env._torso_id, :2].square().sum(dim=-1)
        full_trace.append(torch.column_stack((
            engine.get_root_pos(robot), engine.get_root_vel(robot), engine.get_root_rot(robot),
            engine.get_dof_pos(robot), contact[:, env.foot_ids], nonfoot_force,
            clips.float().mean(dim=-1), done, sole_z, torso_up, env.applied_speed, reward,
            env.fall_flags, engine.get_dof_vel(robot),
        )))
    traces = torch.stack(full_trace, dim=0).cpu().numpy().transpose(1, 0, 2)
    return traces, origins


def window(trace, start, stop, dt, joints, gait, settings, target):
    segment = trace[start:stop]
    if len(segment) != stop - start:
        return None
    native_size = 15 + len(joints)
    result, = describe_windows(segment[:, :native_size], dt, joints, gait)
    flight = ((segment[:, native_size:native_size + 2] > settings["sole_clearance_m"]).all(axis=1)
              & (segment[:, native_size - 5:native_size - 3] < gait["contact_force_n"]).all(axis=1))
    bouts = int(flight[0]) + int(np.count_nonzero(flight[1:] & ~flight[:-1]))
    if result["mode"] == "walk" and flight.sum() > settings["max_walk_flight_frames_per_window"]:
        result["mode"] = "unclassified"
    if result["mode"] == "jog" and (bouts < settings["min_jog_flight_bouts_per_window"]
                                   or flight.mean() < gait["jog"]["no_foot_support_fraction"][0]):
        result["mode"] = "unclassified"
    target_speed, target_mode = target
    mae = float(np.abs(segment[:, 3] - target_speed).mean())
    torso_tilt = np.rad2deg(np.arccos(np.clip(segment[:, native_size + 2], -1, 1)))
    result.update(start_s=start * dt, forward_mae_m_s=mae,
                  flight_fraction=float(flight.mean()), flight_bouts=bouts,
                  torso_tilt_p90_deg=float(np.quantile(torso_tilt, 0.90)),
                  passes=result["mode"] == target_mode and mae <= settings["forward_mae_limit_m_s"])
    return result


def describe_episode(case, trace, origin, dt, joints, gait, protocol):
    settings = protocol["evaluation"]
    steps = gait["max_steps"]
    switch = round(settings["switch_s"] / dt)
    width = round(gait["window_s"] / dt)
    native_size = 15 + len(joints)
    reached = np.flatnonzero(trace[:, native_size - 1])
    count = int(reached[0]) + 1 if len(reached) else steps
    trace = trace[:count]
    assert np.isfinite(trace).all()
    terminal = DoneFlags(int(trace[-1, native_size - 1])).name
    assert terminal in ("TIME", "FAIL")
    flags = int(trace[-1, native_size + 5])
    if terminal == "TIME":
        assert count == steps and flags == 0
    else:
        assert flags != 0

    first, last = SCHEDULES[case["schedule"]]
    target = np.full(steps, case["initial_command_m_s"])
    target[switch:] = case["final_command_m_s"]
    assert np.allclose(trace[:, native_size + 3], target[:count])
    error = np.abs(np.pad(trace[:, 3], (0, steps - count)) - target)
    ranges = [(steps, last)]
    if first != last:
        ranges.insert(0, (switch, first))
    tails = []
    for end, skill in ranges:
        score = protocol["commands_m_s"][skill]
        tail = [
            window(trace, end - i * width, end - (i - 1) * width, dt, joints, gait, settings, (score, skill))
            for i in (2, 1)
        ]
        tails.append(tail)

    latency = None
    if first != last and count > switch:
        stride = round(settings["switch_check_stride_s"] / dt)
        for start in range(switch, count - width + 1, stride):
            candidate = window(trace, start, start + width, dt, joints, gait, settings,
                               (case["final_command_m_s"], last))
            if candidate["passes"]:
                latency = (start + width - switch) * dt
                break
    rotation = Rotation.from_quat(trace[:, 6:10]).as_matrix()
    heading = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    result = {
        **case, "steps": count, "duration_s": count * dt, "terminal": terminal,
        "fall_reasons": [label for mask, label in ((1, "low_pelvis"), (2, "pelvis_tilt"),
                                                   (4, "nonfoot_contact")) if flags & mask],
        "success": terminal == "TIME" and all(
            item is not None and item["passes"] for pair in tails for item in pair
        ),
        "tail_windows": tails, "switch_latency_s": latency,
        "reached_switch": first != last and count > switch,
        "mae_vx_full_m_s": float(error.mean()),
        "mae_vx_alive_m_s": float(error[:count].mean()),
        "mae_vx_after_switch_m_s": float(error[switch:].mean()) if first != last else None,
        "mae_vy_full_m_s": float(np.abs(trace[:, 4]).sum() / steps),
        "mae_heading_alive_deg": float(np.abs(np.rad2deg(heading)).mean()),
        "world_forward_m": float(trace[-1, 0] - origin[0]),
        "world_lateral_m": float(trace[-1, 1] - origin[1]),
        "target_clip_fraction": float(trace[:, native_size - 2].mean()),
    }
    return result, trace


def group_stats(episodes):
    complete = [case for case in episodes if case["terminal"] == "TIME"]
    latencies = [case["switch_latency_s"] for case in episodes if case["switch_latency_s"] is not None]
    windows = [window for case in complete for window in case["tail_windows"][-1]]
    duration = [case["duration_s"] for case in episodes]
    return {
        "episodes": len(episodes), "successes": sum(case["success"] for case in episodes),
        "completed": len(complete), "falls": len(episodes) - len(complete),
        "mean_duration_s": mean(duration),
        "mean_mae_vx_full_m_s": mean(case["mae_vx_full_m_s"] for case in episodes),
        "mean_mae_vx_alive_m_s": mean(case["mae_vx_alive_m_s"] for case in episodes),
        "mean_mae_vy_full_m_s": mean(case["mae_vy_full_m_s"] for case in episodes),
        "mean_mae_heading_alive_deg": mean(case["mae_heading_alive_deg"] for case in episodes),
        "mean_forward_m": mean(case["world_forward_m"] for case in episodes),
        "mean_abs_lateral_m": mean(abs(case["world_lateral_m"]) for case in episodes),
        "completed_tail_window_modes": dict(Counter(window["mode"] for window in windows)),
        "completed_tail_mean_local_vx_m_s": mean(window["median_forward_m_s"] for window in windows) if windows else None,
        "completed_tail_mean_flight_fraction": mean(window["flight_fraction"] for window in windows) if windows else None,
        "completed_tail_mean_support_switches": mean(window["foot_support_switches"] for window in windows) if windows else None,
        "completed_tail_mean_forward_mae_m_s": mean(window["forward_mae_m_s"] for window in windows) if windows else None,
        "switch_attempts": sum(case["reached_switch"] for case in episodes),
        "switch_acquisitions": len(latencies),
        "switch_latency_mean_s": mean(latencies) if latencies else None,
    }


@torch.no_grad()
def compare_reference(env, episodes, traces, joints, audit, dt, gait):
    width = 2 * round(gait["window_s"] / dt)
    library = env._motion_lib
    device = env._device
    result = {}
    for skill in ("walk", "jog"):
        matches = [ep for ep in episodes if ep["schedule"] == skill and ep["terminal"] == "TIME"]
        if not matches:
            result[skill] = {"completed_episodes": 0}
            continue
        poses = np.concatenate([traces[ep["id"]][-width:] for ep in matches])
        sources = []
        for i, clip in enumerate(audit["clips"]):
            assert Path(library.get_motion_file(i)).name == clip["motion_file"]
            if clip["skill"] == skill:
                ids = torch.full((width,), i, dtype=torch.long, device=device)
                phase = (torch.arange(width, device=device) + .5) / width
                _, _, velocity, _, rot, dof_vel = library.calc_motion_frame(ids, phase * library.get_motion_length(ids))
                sources.append(torch.column_stack((velocity[:, 0], library.joint_rot_to_dof(rot), dof_vel)).cpu().numpy())
        reference = np.concatenate(sources)
        offset = 15 + len(joints) + 6
        observed = np.column_stack((poses[:, 3], poses[:, 10:10 + len(joints)], poses[:, offset:]))
        distances = [wasserstein_distance(actual, expert) for actual, expert in zip(observed.T, reference.T, strict=True)]
        result[skill] = {
            "completed_episodes": len(matches), "expert_samples": len(reference),
            "velocity_w1_m_s": distances[0], "mean_joint_position_w1_rad": mean(distances[1:1 + len(joints)]),
            "mean_joint_velocity_w1_rad_s": mean(distances[1 + len(joints):]),
            "joint_position_w1_rad": dict(zip(joints, distances[1:1 + len(joints)], strict=True)),
            "joint_velocity_w1_rad_s": dict(zip(joints, distances[1 + len(joints):], strict=True)),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--gait", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("fixed", "held_out"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    gait = json.loads(args.gait.read_text())
    window_gait = {**gait, "burn_in_s": 0.0}
    env_config, agent_config = prepare_configs(args.data, protocol, args.variant)
    audit = json.loads((args.data / "audit.json").read_text())
    joints = audit["joint_names"]
    device = "cuda:0"
    mp_util.init(0, 1, device, None)
    util.set_rand_seed(protocol["evaluation"][f"{args.split}_seed"])
    phases = protocol["evaluation"][f"{args.split}_phases_per_initial_skill"]
    env, agent = build_env_agent(env_config, agent_config, phases * 2 * len(SCHEDULES), device)
    agent.load(str(args.checkpoint))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    dt = env._engine.get_timestep()
    cases = build_cases(env, audit, protocol, args.split)
    traces, origins = roll_out(env, agent, cases, gait["max_steps"], round(protocol["evaluation"]["switch_s"] / dt),
                               sole_geometry(env_config["char_file"], device))
    episodes, records = [], {}
    for case, trace, origin in zip(cases, traces, origins, strict=True):
        episode, record = describe_episode(case, trace, origin, dt, joints, window_gait, protocol)
        episodes.append(episode)
        records[episode["id"]] = record
    groups = {}
    for schedule in SCHEDULES:
        batch = [episode for episode in episodes if episode["schedule"] == schedule]
        groups[schedule] = group_stats(batch)
        for skill in ("walk", "jog"):
            groups[f"{skill}_init/{schedule}"] = group_stats([e for e in batch if e["initial_skill"] == skill])
        print(schedule, groups[schedule], flush=True)
    columns = [
        "x", "y", "z", "vx_world", "vy_world", "vz_world", "qx", "qy", "qz", "qw", *joints,
        "left_sole_force_n", "right_sole_force_n", "nonfoot_max_force_n", "clipped_fraction", "done",
        "left_sole_z_m", "right_sole_z_m", "torso_up_z", "command_m_s", "task_reward", "fall_flags",
        *(f"{name}_velocity_rad_s" for name in joints),
    ]
    assert len(columns) == traces.shape[-1]
    data = {
        "variant": args.variant, "split": args.split,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "dt": dt, "protocol": protocol, "gait": gait, "columns": columns, "groups": groups,
        "style_of_completed_constant_command_episodes": compare_reference(env, episodes, records, joints, audit, dt, gait),
        "episodes": episodes,
    }
    np.savez_compressed(args.output.with_suffix(".npz"), **records)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
