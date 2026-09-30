"""Load an rsl_rl checkpoint as a deterministic (mean-action) policy on an Isaac Lab env."""

from __future__ import annotations

import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner


def load_policy(env, task: str, checkpoint: str):
    """Returns (wrapped_env, policy_fn, policy_module). policy_fn(obs) -> mean actions."""
    import importlib.metadata as metadata

    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint, map_location=agent_cfg.device)
    module = runner.get_inference_policy(device=env.unwrapped.device)

    def policy(obs) -> torch.Tensor:
        return module(obs)  # MLPModel.forward: deterministic output unless stochastic_output=True

    return wrapped, policy, module
