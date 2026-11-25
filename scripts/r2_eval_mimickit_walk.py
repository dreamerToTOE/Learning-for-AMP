"""Evaluate a G1 checkpoint using MimicKit's deterministic policy and ground contacts."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def evaluate_episode(env, agent, seed, max_steps, contact_threshold):
    from envs.base_env import DoneFlags
    from util.util import set_rand_seed

    set_rand_seed(seed)
    obs, info = env.reset()
    initial_sha256 = hashlib.sha256(obs[0].cpu().numpy().tobytes()).hexdigest()
    engine = env._engine
    robot = env._get_char_id()
    names = engine.get_obj_body_names(robot)
    foot_ids = [names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
    nonfoot_ids = [index for index in range(len(names)) if index not in foot_ids]
    start = engine.get_root_pos(robot)[0].cpu().numpy().copy()
    bound_low = env._action_bound_low
    bound_high = env._action_bound_high

    root_positions = []
    foot_contacts = []
    nonfoot_force = []
    action_peak = 0.0
    out_of_bounds = 0
    for step in range(1, max_steps + 1):
        action, _ = agent._decide_action(obs, info)
        obs, _, done, info = env.step(action)
        pos = engine.get_root_pos(robot)[0].cpu().numpy().copy()
        force = engine.get_ground_contact_forces(robot)[0].norm(dim=-1).cpu().numpy()
        if not torch.isfinite(action).all() or not torch.isfinite(obs).all():
            raise ValueError(f"Non-finite observation/action at seed {seed}, step {step}")
        if not np.isfinite(pos).all() or not np.isfinite(force).all():
            raise ValueError(f"Non-finite root position/contact at seed {seed}, step {step}")
        root_positions.append(pos)
        foot_contacts.append(force[foot_ids] >= contact_threshold)
        nonfoot_force.append(float(force[nonfoot_ids].max()))
        action_peak = max(action_peak, action.abs().max().item())
        out_of_bounds += torch.count_nonzero((action < bound_low) | (action > bound_high)).item()
        if done[0].item() != DoneFlags.NULL.value:
            break

    root_positions = np.asarray(root_positions)
    contacts = np.asarray(foot_contacts)
    exclusive = contacts[contacts.sum(axis=1) == 1, 0]
    switches = int(np.count_nonzero(exclusive[1:] != exclusive[:-1]))
    elapsed = step * engine.get_timestep()
    forward = float(root_positions[-1, 0] - start[0])
    return {
        "seed": seed,
        "initial_observation_sha256": initial_sha256,
        "steps": step,
        "terminal": DoneFlags(done[0].item()).name,
        "duration_s": elapsed,
        "forward_m": forward,
        "forward_m_per_s": forward / elapsed,
        "lateral_m": float(root_positions[-1, 1] - start[1]),
        "pelvis_height_p05_m": float(np.quantile(root_positions[:, 2], 0.05)),
        "foot_support_switches": switches,
        "nonfoot_contact_steps": int(np.count_nonzero(np.asarray(nonfoot_force) >= contact_threshold)),
        "nonfoot_peak_force_n": max(nonfoot_force),
        "max_abs_raw_joint_target_rad": action_peak,
        "target_clip_fraction": out_of_bounds / (step * action.numel()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--video", type=Path, help="Save an MP4 for a single evaluation seed.")
    parser.add_argument("--env-config", default="data/envs/amp_g1_env.yaml")
    parser.add_argument("--agent-config", default="data/agents/amp_g1_agent.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.video and len(args.seeds) != 1:
        parser.error("--video requires one evaluation seed")

    upstream = args.mimickit.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    video_path = args.video.resolve() if args.video else None
    os.chdir(upstream)
    sys.path.insert(0, str(upstream / "mimickit"))
    from envs.env_builder import build_env
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import mp_util

    mp_util.init(0, 1, args.device, None)
    env = build_env(
        args.env_config, "data/engines/newton_engine.yaml", 1, args.device,
        visualize=False, record_video=video_path is not None,
    )
    agent = build_agent(args.agent_config, env, args.device)
    agent.load(str(checkpoint))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    max_steps = round(env._episode_length / env._engine.get_timestep())
    contact_threshold = 5.0
    with torch.no_grad():
        episodes = [evaluate_episode(env, agent, seed, max_steps, contact_threshold) for seed in args.seeds]

    if video_path:
        recording = env.record_diagnostics()["sim_recording"]
        frames = recording.get_frames()
        if len(frames) != episodes[0]["steps"] or not any(np.any(frame) for frame in frames):
            raise RuntimeError("The Newton recorder did not capture a complete visible episode")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        recording.save(str(video_path))

    result = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "env_config": args.env_config,
        "agent_config": args.agent_config,
        "max_episode_steps": max_steps,
        "contact_threshold_n": contact_threshold,
        "episodes": episodes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {len(episodes)} episodes to {output}")


if __name__ == "__main__":
    main()
