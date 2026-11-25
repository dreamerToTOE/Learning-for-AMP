"""Replay diagnostics for the frozen Humanoid28 speed-task policies."""

from hashlib import sha256
import torch
import yaml
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.utils.model_instantiators.torch import deterministic_model

from r1_speed_task import RULE_NAMES, style_reward_from_logits


class RewardAudit:
    def __init__(self, env, style_references):
        self.env = env
        self.scorers = {}
        self.references = {}
        for label, path in style_references:
            config = yaml.safe_load((path.parents[1] / "params" / "agent.yaml").read_text())
            model_cfg = config["models"]["discriminator"].copy()
            if model_cfg.pop("class") != "DeterministicMixin":
                raise ValueError("expected a deterministic skrl AMP discriminator")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = deterministic_model(
                observation_space=env.amp_observation_space, action_space=env.single_action_space,
                device=env.device, **model_cfg,
            )
            model.init_state_dict("discriminator")
            model.load_state_dict(checkpoint["discriminator"])
            model.eval()
            scaler = RunningStandardScaler(
                size=env.amp_observation_space, device=env.device,
                **(config["agent"].get("amp_state_preprocessor_kwargs") or {}),
            )
            scaler.load_state_dict(checkpoint["amp_state_preprocessor"])
            scaler.eval()
            scale = config["agent"]["discriminator_reward_scale"]
            self.scorers[label] = (model, scaler, scale)
            self.references[label] = {
                "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
                "scale": scale,
                "task_reward_weight": config["agent"]["task_reward_weight"],
                "style_reward_weight": config["agent"]["style_reward_weight"],
            }
        self.reset()

    def reset(self):
        self.rows = []
        self.pre_reset_amp_obs = []
        self.returned_amp_obs = []

    def record(self, info):
        env = self.env
        metrics = env._r1_speed_metrics
        tilt_deg = torch.rad2deg(torch.acos(env.audit_up_z.clamp(-1, 1)))
        row = torch.stack((
            env.step_positions[env.ref_body_index, 2],
            tilt_deg[0],
            metrics["torso_vx"][0],
            metrics["speed_bonus"][0],
            env.audit_original_alive[0],
            metrics["alive_reward"][0],
            metrics["task_reward"][0],
        ))
        if env.use_ground_sensor:
            row = torch.cat((row, env.ground_forces_n[0]))
        self.rows.append(row)
        if self.scorers:
            self.pre_reset_amp_obs.append(env.audit_amp_obs[0])
            self.returned_amp_obs.append(info["amp_obs"][0].clone())

    def finish(self):
        env = self.env
        labels = [
            "torso_height_m", "torso_tilt_deg", "torso_vx_mps", "speed_bonus_before_gate",
            "legacy_alive", "applied_alive", "applied_task_reward",
        ]
        if env.use_ground_sensor:
            labels.extend("ground_normal_force_n/" + name for name in env.ground_body_names)
        traces = dict(zip(labels, torch.stack(self.rows, dim=1).cpu().tolist()))
        for label, (model, scaler, scale) in self.scorers.items():
            for state_label, states in (
                ("pre_reset", self.pre_reset_amp_obs),
                ("skrl_input", self.returned_amp_obs),
            ):
                logits, _ = model.compute({"states": scaler(torch.stack(states))}, role="discriminator")
                traces[f"{label}/{state_label}_style_reward"] = style_reward_from_logits(logits, scale).flatten().cpu().tolist()
        return {
            "traces": traces,
            "original_terminated": bool(env.original_terminated[0].item()),
            "new_rule_flags_at_end": [
                name for name, active in zip(RULE_NAMES, env.rule_terminated[0].tolist()) if active
            ],
        }
