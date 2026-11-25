"""Train either discriminator variant through MimicKit's native AMP loop."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from envs.amp_env import AMPEnv
from envs.base_env import DoneFlags
from learning.base_agent import AgentMode
from util import mp_util, util

from r3_speed_amp import VARIANTS, build_env_agent, prepare_configs, save_configs


def parameter_hashes(agent):
    model = agent._model
    groups = {"actor": model.get_actor_params(), "critic": model.get_critic_params(),
              "discriminator": model.get_disc_params()}
    result = {}
    for key, parameters in groups.items():
        digest = hashlib.sha256()
        for tensor in parameters:
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        result[key] = digest.hexdigest()
    return result


@torch.no_grad()
def check_command_path(env, agent):
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    env.auto_commands = False
    obs, info = env.reset()
    obs_without_command = obs[:, :-1].clone()
    walk, jog = env._speeds
    env.set_eval_speed(walk.expand_as(env.applied_speed))
    assert torch.all(obs[:, -1] == walk)
    assert torch.equal(obs[:, :-1], obs_without_command)
    action, _ = agent._decide_action(obs, info)

    update_task = env._update_task
    try:
        env._update_task = lambda: env._tar_speed.copy_(jog)
        after, reward, done, after_info = env.step(action)
    finally:
        env._update_task = update_task
        env.auto_commands = True
    assert torch.all(after[:, -1] == jog) and torch.all(env.applied_speed == walk)
    assert reward.isfinite().all() and action.isfinite().all()
    assert torch.all(reward[env.fall_flags != 0] == 0)
    assert torch.equal(done == DoneFlags.FAIL.value, env.fall_flags != 0)

    motion_ids = torch.arange(env._motion_lib.get_num_motions(), device=env._device)
    times = 0.5 * env._motion_lib.get_motion_length(motion_ids)
    native = AMPEnv._compute_disc_obs_demo(env, motion_ids, times)
    demos = env._compute_disc_obs_demo(motion_ids, times)
    if env.condition_discriminator:
        assert torch.equal(demos[..., :-1], native)
        assert torch.equal(demos[..., -1], env.demo_speed)
        assert torch.all(after_info["disc_obs"][..., -1] == walk)
    else:
        assert torch.equal(demos, native)
    return {
        "actor_obs_size": obs.shape[-1],
        "disc_obs_size": demos.shape[-1],
        "switch_step_command_m_s": float(env.applied_speed[0]),
        "next_obs_command_m_s": float(after[0, -1]),
        "demo_commands_m_s": env.demo_speed.tolist(),
        "preflight_failed_envs": int((done == DoneFlags.FAIL.value).sum().item()),
    }


def check_replay(agent):
    env = agent._env
    if not env.condition_discriminator:
        return {}
    latest = agent._exp_buffer.get_data_flat("disc_obs")[..., -1]
    applied = agent._exp_buffer.get_data_flat("obs")[..., -1]
    assert torch.equal(latest, applied), "Policy motion has the wrong command"
    replay = agent._disc_buffer.sample(2048)["disc_obs"]
    assert torch.isin(replay[..., -1], env._speeds).all()
    assert torch.unique(replay[..., -1]).numel() == len(env._speeds)
    sample = replay[:1].detach().requires_grad_()
    gradient, = torch.autograd.grad(agent._model.eval_disc(sample).sum(), sample)
    assert torch.all(gradient[..., -1] == 0) and torch.any(gradient[..., :-1] != 0)
    return {
        "last_rollout_command_counts": [(latest == speed).sum().item() for speed in env._speeds],
        "replay_command_counts": [(replay[..., -1] == speed).sum().item() for speed in env._speeds],
        "gradient_penalty_excludes_command": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    train = protocol["training"]
    mp_util.init(0, 1, "cuda:0", None)
    util.set_rand_seed(train["seed"])
    env_config, agent_config = prepare_configs(args.data, protocol, args.variant)
    env, agent = build_env_agent(env_config, agent_config, train["environments"], "cuda:0")
    save_configs(env_config, agent_config, args.output)
    initial = parameter_hashes(agent)
    (args.output / "initial_parameters.json").write_text(json.dumps(initial, indent=2) + "\n")
    preflight = check_command_path(env, agent) if args.smoke else {}
    util.set_rand_seed(train["seed"])
    steps = train["smoke_steps"] if args.smoke else train["formal_steps"]
    assert steps % (train["environments"] * agent_config["steps_per_iter"]) == 0
    agent.train_model(max_samples=steps, out_dir=str(args.output), save_int_models=True, logger_type="txt")
    if args.smoke:
        final = parameter_hashes(agent)
        updated = {key: initial[key] != final[key] for key in initial}
        assert all(updated.values())
        preflight.update(check_replay(agent), parameters_updated=updated)
        (args.output / "smoke_checks.json").write_text(json.dumps(preflight, indent=2) + "\n")


if __name__ == "__main__":
    main()
