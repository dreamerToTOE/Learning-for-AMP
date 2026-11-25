"""Project-local Humanoid28 speed task with an opt-in upright walking rule."""

import json
import math

import torch
from pxr import Usd

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv, compute_obs


TARGET_VX_MPS = 1.5
SIGMA_VX_MPS = 0.5
MIN_TORSO_HEIGHT_M = 0.80
MAX_TORSO_TILT_DEG = 45.0
NON_FOOT_CONTACT_FORCE_N = 10.0
INVALID_STEPS_TO_TERMINATE = 3
RULE_NAMES = ("torso_too_low", "torso_tilt", "non_foot_ground_contact")
FOOT_NAMES = {"left_foot", "right_foot"}


def task_rule_parameters():
    return {
        "target_vx_mps": TARGET_VX_MPS,
        "speed_sigma_mps": SIGMA_VX_MPS,
        "min_torso_height_m": MIN_TORSO_HEIGHT_M,
        "max_torso_tilt_deg": MAX_TORSO_TILT_DEG,
        "non_foot_contact_force_n": NON_FOOT_CONTACT_FORCE_N,
        "consecutive_invalid_steps": INVALID_STEPS_TO_TERMINATE,
    }


def torso_up_z(quaternion):
    # Isaac Lab uses wxyz quaternions. This is the world-Z projection of torso +Z.
    return 1 - 2 * quaternion[..., 1:3].square().sum(dim=-1)


def style_reward_from_logits(logits, scale):
    return -torch.log((1 - 1 / (1 + torch.exp(-logits))).clamp_min(0.0001)) * scale


def install_training_style_log(agent, path):
    """Read the style forward already performed by skrl 1.4.3 during each AMP update."""
    original = agent.discriminator.act
    path.parent.mkdir(parents=True, exist_ok=True)
    updates = 0

    def act(inputs, role=""):
        nonlocal updates
        result = original(inputs, role)
        # skrl's style calculation uses the entire (time, env, feature) rollout.
        if role == "discriminator" and inputs["states"].ndim == 3 and not torch.is_grad_enabled():
            raw_task = agent.memory.get_tensor_by_name("rewards")
            style = style_reward_from_logits(result[0], agent._discriminator_reward_scale)
            if raw_task.numel() != style.numel():
                raise RuntimeError("unexpected skrl AMP style reward sample count")
            updates += 1
            task_mean, style_mean = torch.stack((raw_task.mean(), style.mean())).tolist()
            task = task_mean * agent._task_reward_weight
            style_weighted = style_mean * agent._style_reward_weight
            record = {
                "update": updates, "samples": raw_task.numel(), "task_mean": task_mean, "style_mean": style_mean,
                "weighted_task_mean": task, "weighted_style_mean": style_weighted, "combined_mean": task + style_weighted,
            }
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return result

    agent.discriminator.act = act


class SpeedTrackingWalkEnv(HumanoidAmpEnv):
    def __init__(self, cfg, *, upright_task=False, ground_contacts=False, reward_audit=False, **kwargs):
        self.upright_task = upright_task
        self.reward_audit = reward_audit
        self.ground_contact_sensor = None
        self.use_ground_sensor = ground_contacts or upright_task
        if self.use_ground_sensor:
            cfg.robot.spawn.activate_contact_sensors = True
        super().__init__(cfg, **kwargs)
        self.invalid_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        if self.use_ground_sensor:
            sensor_names = self.ground_contact_sensor.body_names
            if set(sensor_names) != set(self.robot.body_names) or self.ground_contact_sensor.contact_physx_view.filter_count != 1:
                raise RuntimeError("expected one ground filter and all Humanoid28 rigid links")
            self.ground_body_names = tuple(sensor_names)
            self.non_foot_indices = [index for index, name in enumerate(sensor_names) if name not in FOOT_NAMES]

    def _setup_scene(self):
        super()._setup_scene()
        if self.use_ground_sensor:
            ground = self.sim.stage.GetPrimAtPath("/World/ground")
            plane, = [str(prim.GetPath()) for prim in Usd.PrimRange(ground) if prim.GetTypeName() == "Plane"]
            self.ground_contact_sensor = ContactSensor(
                ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", filter_prim_paths_expr=[plane]),
            )
            self.scene.sensors["r1_ground_contacts"] = self.ground_contact_sensor

    def _get_dones(self):
        died, timeout = super()._get_dones()
        if self.upright_task or self.reward_audit:
            robot = self.robot.data
            torso = self.ref_body_index
            height = robot.body_pos_w[:, torso, 2]
            up_z = torso_up_z(robot.body_quat_w[:, torso])
            self.ground_forces_n = None
            ground_contact = torch.zeros_like(died)
            if self.use_ground_sensor:
                self.ground_forces_n = self.ground_contact_sensor.data.force_matrix_w[:, :, 0, 2]
                ground_contact = (self.ground_forces_n[:, self.non_foot_indices] >= NON_FOOT_CONTACT_FORCE_N).any(dim=1)
                if self.reward_audit:
                    self.ground_forces_n = self.ground_forces_n.clone()
            self.rule_invalid = torch.stack(
                (height < MIN_TORSO_HEIGHT_M, up_z < math.cos(math.radians(MAX_TORSO_TILT_DEG)), ground_contact),
                dim=1,
            )
            self.invalid_steps = torch.where(self.rule_invalid.any(dim=1), self.invalid_steps + 1, 0)
            self.rule_terminated = self.rule_invalid & (self.invalid_steps[:, None] >= INVALID_STEPS_TO_TERMINATE)
            self.original_terminated = died
            if self.reward_audit:
                # Record before DirectRLEnv replaces the terminal state with a reset observation.
                self.audit_up_z = up_z
                self.audit_amp_obs = torch.cat(
                    (
                        compute_obs(
                            robot.joint_pos, robot.joint_vel, robot.body_pos_w[:, torso], robot.body_quat_w[:, torso],
                            robot.body_lin_vel_w[:, torso], robot.body_ang_vel_w[:, torso],
                            robot.body_pos_w[:, self.key_body_indexes],
                        ),
                        self.amp_observation_buffer[:, :-1].flatten(start_dim=1),
                    ),
                    dim=1,
                )
            if self.upright_task:
                died = died | self.rule_terminated.any(dim=1)
        return died, timeout

    def _get_rewards(self):
        torso = self.ref_body_index
        vx = self.robot.data.body_lin_vel_w[:, torso, 0].clone()
        height = self.robot.data.body_pos_w[:, torso, 2]
        speed_error = vx - TARGET_VX_MPS
        speed_bonus = torch.exp(-0.5 * (speed_error / SIGMA_VX_MPS).square())
        original_alive = height >= self.cfg.termination_height
        alive = original_alive
        if self.upright_task:
            alive = alive & ~self.rule_invalid.any(dim=1)
        alive = alive.to(vx.dtype)
        task_reward = alive * (1.0 + speed_bonus)
        self._r1_speed_metrics = {
            "alive_reward": alive,
            "speed_bonus": speed_bonus,
            "task_reward": task_reward,
            "torso_vx": vx,
            "speed_abs_error": speed_error.abs(),
        }
        if self.upright_task:
            self._r1_speed_metrics["invalid_step"] = ~alive.bool()
        if self.reward_audit:
            self.audit_original_alive = original_alive
        return task_reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self, "invalid_steps"):
            self.invalid_steps[env_ids if env_ids is not None else slice(None)] = 0
