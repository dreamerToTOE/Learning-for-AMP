"""Reuse MimicKit steering and AMP for the G1 two-speed comparison."""

import hashlib
import json
import math
from pathlib import Path

import torch
import yaml

from envs import amp_env, env_builder
from envs.base_env import DoneFlags
from envs.task_steering_env import TaskSteeringEnv, compute_steering_reward
from learning.amp_agent import AMPAgent
from learning.amp_model import AMPModel


FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
ENGINE_CONFIG = "data/engines/newton_engine.yaml"
VARIANTS = ("ordinary", "conditional")


class SpeedCommandEnv(TaskSteeringEnv):
    def __init__(self, config, engine_config, num_envs, device):
        self.condition_discriminator = config["condition_discriminator"]
        self.auto_commands = True
        self._speeds = torch.tensor(config["command_speeds"], dtype=torch.float, device=device)
        self._demo_commands_by_file = config["demo_commands_by_file"]
        self._min_height = config["fall"]["min_pelvis_height_m"]
        self._min_pelvis_up = math.cos(math.radians(config["fall"]["max_pelvis_tilt_deg"]))
        self._max_nonfoot_force_sq = config["fall"]["nonfoot_contact_n"] ** 2
        super().__init__(config, engine_config, num_envs, device, visualize=False)

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self.applied_speed = self._tar_speed.clone()
        names = self._engine.get_obj_body_names(self._get_char_id())
        self.foot_ids = [names.index(foot) for foot in FEET]
        self._torso_id = names.index("torso_link")
        self._nonfoot_ids = [i for i in range(len(names)) if i not in self.foot_ids]
        self.fall_flags = torch.zeros_like(self._tar_speed, dtype=torch.int)
        self.demo_speed = torch.tensor([
            self._demo_commands_by_file[Path(self._motion_lib.get_motion_file(i)).name]
            for i in range(self._motion_lib.get_num_motions())
        ], dtype=torch.float, device=self._device)

    def _build_data_buffers(self):
        super()._build_data_buffers()
        if self.condition_discriminator:
            # MimicKit writes motion features to this view. Its info and replay retain the extra column.
            self._disc_obs_buf = self._disc_obs_buf[..., :-1]

    def _compute_obs(self, env_ids=None):
        obs = amp_env.AMPEnv._compute_obs(self, env_ids)
        speed = self._tar_speed if env_ids is None else self._tar_speed[env_ids]
        return torch.cat((obs, speed.unsqueeze(-1)), dim=-1)

    def _reset_tar(self, env_ids):
        index = torch.randint(len(self._speeds), (len(env_ids),), device=self._device)
        self._tar_speed[env_ids] = self._speeds[index]

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        self.applied_speed[env_ids] = self._tar_speed[env_ids]
        self.fall_flags[env_ids] = 0

    def _update_task(self):
        if self.auto_commands:
            super()._update_task()

    def set_eval_speed(self, speed):
        assert not self.auto_commands
        self._tar_speed.copy_(speed)
        self._obs_buf[..., -1] = self._tar_speed

    def _pre_physics_step(self, actions):
        # TaskSteeringEnv samples the next command before post-step rewards are evaluated.
        self.applied_speed.copy_(self._tar_speed)
        super()._pre_physics_step(actions)

    def _update_reward(self):
        robot = self._get_char_id()
        root_pos = self._engine.get_root_pos(robot)
        root_rot = self._engine.get_root_rot(robot)
        pelvis_up = 1.0 - 2.0 * root_rot[:, :2].square().sum(dim=-1)
        nonfoot = self._engine.get_ground_contact_forces(robot)[:, self._nonfoot_ids]
        contact = nonfoot.square().sum(dim=-1).amax(dim=-1) > self._max_nonfoot_force_sq
        low = root_pos[:, 2] < self._min_height
        tilted = pelvis_up < self._min_pelvis_up
        self.fall_flags[:] = low.int() + 2 * tilted.int() + 4 * contact.int()

        reward = compute_steering_reward(
            root_pos, self._prev_root_pos, root_rot, self._tar_dir, self.applied_speed, self._face_dir,
            self._engine.get_timestep(), self._reward_steering_vel_scale,
            self._reward_steering_tar_w, self._reward_steering_face_w,
        )
        self._reward_buf[:] = reward.masked_fill(self.fall_flags != 0, 0.0)

    def _update_done(self):
        self._done_buf.fill_(DoneFlags.NULL.value)
        self._done_buf.masked_fill_(self._time_buf >= self._episode_length, DoneFlags.TIME.value)
        self._done_buf.masked_fill_(self.fall_flags != 0, DoneFlags.FAIL.value)

    def _update_disc_obs(self, env_ids=None):
        super()._update_disc_obs(env_ids)
        if self.condition_discriminator:
            rows = slice(None) if env_ids is None else env_ids
            self._info["disc_obs"][rows, -1] = self.applied_speed[rows]

    def _compute_disc_obs_demo(self, motion_ids, motion_times):
        obs = super()._compute_disc_obs_demo(motion_ids, motion_times)
        if self.condition_discriminator:
            obs = torch.cat((obs, self.demo_speed[motion_ids].unsqueeze(-1)), dim=-1)
        return obs


class SpeedAMPModel(AMPModel):
    def eval_disc(self, disc_obs):
        # Regularize gradients on motion features only. The command is an observed condition.
        disc_obs = torch.cat((disc_obs[..., :-1], disc_obs[..., -1:].detach()), dim=-1)
        return super().eval_disc(disc_obs)


class SpeedAMPAgent(AMPAgent):
    def _build_model(self, config):
        if self._env.condition_discriminator:
            self._model = SpeedAMPModel(config["model"], self._env)
        else:
            super()._build_model(config)

    def _compute_rewards(self):
        task_mean = self._exp_buffer.get_data_flat("reward").mean()
        return {**super()._compute_rewards(), "task_reward_mean": task_mean}


def prepare_configs(data, protocol, variant):
    env = env_builder.load_config(data / "amp_g1_walk_jog_env.yaml")
    agent = env_builder.load_config(data / "amp_g1_walk_jog_agent.yaml")
    audit = json.loads((data / "audit.json").read_text())
    commands = protocol["commands_m_s"]
    speeds = [commands["walk"], commands["jog"]]
    start, stop = protocol["command_duration_s"]
    demo_commands = {clip["motion_file"]: commands[clip["skill"]] for clip in audit["clips"]}
    dataset = env_builder.load_config(Path(env["motion_file"]))
    paths = [Path(row["file"]) for row in dataset["motions"]]
    if set(path.name for path in paths) != demo_commands.keys():
        raise ValueError("The trainer motion library must match the frozen HOLD manifest")
    for path in paths:
        expected = next(clip["motion_sha256"] for clip in audit["clips"] if clip["motion_file"] == path.name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Changed prepared motion: {path}")
    env.update(
        env_name="g1_speed_steering",
        rand_tar_dir=False, rand_face_dir=False,
        tar_speed_min=min(speeds), tar_speed_max=max(speeds),
        tar_change_time_min=start, tar_change_time_max=stop,
        command_speeds=speeds, demo_commands_by_file=demo_commands,
        contact_bodies=list(FEET), fall=protocol["fall"],
        condition_discriminator=variant == "conditional",
        **protocol["steering_reward"],
    )
    agent.update(
        task_reward_weight=protocol["task_reward_weight"], disc_reward_weight=protocol["disc_reward_weight"],
        iters_per_output=protocol["training"]["iters_per_output"],
    )
    return env, agent


def build_env_agent(env_config, agent_config, num_envs, device):
    engine = env_builder.override_engine_config(env_config["engine"], env_builder.load_config(ENGINE_CONFIG))
    env = SpeedCommandEnv(env_config, engine, num_envs, device)
    agent = SpeedAMPAgent(agent_config, env, device)
    return env, agent


def save_configs(env_config, agent_config, output):
    (output / "env_config.yaml").write_text(yaml.safe_dump(env_config, sort_keys=False))
    (output / "agent_config.yaml").write_text(yaml.safe_dump(agent_config, sort_keys=False))
