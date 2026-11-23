"""Optional local action-range correction for R1 controlled experiments."""

from __future__ import annotations


def apply_action_range_patch() -> None:
    """Clamp policy actions and map [-1, 1] to the full joint-limit interval.

    The upstream task stores the full soft-limit width as ``action_scale``.
    This local patch uses half of that width, which is the affine mapping from
    normalized actions to joint limits. It is opt-in and does not modify the
    isolated Isaac Lab checkout.
    """

    import torch
    from isaaclab_tasks.direct.humanoid_amp.humanoid_amp_env import HumanoidAmpEnv

    if getattr(HumanoidAmpEnv, "_r1_action_range_patch", False):
        return

    original_init = HumanoidAmpEnv.__init__

    def patched_init(self, cfg, render_mode=None, **kwargs):
        original_init(self, cfg, render_mode, **kwargs)
        self.action_scale = 0.5 * self.action_scale

    def patched_pre_physics_step(self, actions):
        self.actions = torch.clamp(actions, -1.0, 1.0).clone()

    HumanoidAmpEnv.__init__ = patched_init
    HumanoidAmpEnv._pre_physics_step = patched_pre_physics_step
    HumanoidAmpEnv._r1_action_range_patch = True
    print("[R1] Applied local action-range correction: clamp [-1, 1], half joint-limit width", flush=True)
