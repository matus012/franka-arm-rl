"""Env sanity check used by tests/test_sim.py: reset/step shapes + dtypes, every reward term finite and
inside its expected range under random actions. Prints one JSON line prefixed with CHECK_ENV=.

Usage: uv run python scripts/check_env.py --task LiftRL-Own-v0 --num_envs 8 --steps 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import torch

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.physics import assert_run_preconditions

# raw (unweighted) value range per reward term
RANGES = {
    "reach": (0.0, 1.0), "reaching_object": (0.0, 1.0),
    "grasp": (0.0, 1.0), "lift": (0.0, 1.0), "lifting_object": (0.0, 1.0),
    "track_coarse": (0.0, 1.0), "track_fine": (0.0, 1.0), "track_precise": (0.0, 1.0),
    "object_goal_tracking": (0.0, 1.0), "object_goal_tracking_fine_grained": (0.0, 1.0),
    "action_rate": (0.0, 1.0e3), "joint_vel": (0.0, 1.0e4),
    "table_slam": (0.0, 1.0), "jam": (0.0, 1.0),
    "staged": (0.0, 4.9 + 1e-4), "hold_bonus": (0.0, 1.0), "low_fast": (0.0, 20.0), "table_hit": (0.0, 1.0),
}

p = argparse.ArgumentParser()
p.add_argument("--task", default="LiftRL-Own-v0")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--steps", type=int, default=60)
args = p.parse_args()

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = "cuda:0"
assert_run_preconditions(env_cfg)

with launch_simulation(env_cfg, {"headless": True}):
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset(seed=0)
    res: dict = {"obs_shape": list(obs["policy"].shape), "obs_dtype": str(obs["policy"].dtype),
                 "action_dim": int(u.action_manager.total_action_dim), "obs_terms": u.observation_manager.active_terms["policy"],
                 "obs_term_dims": [list(d) for d in u.observation_manager.group_obs_term_dim["policy"]]}
    lo_hi = {n: [float("inf"), float("-inf")] for n in u.reward_manager.active_terms}
    finite = True
    for t in range(args.steps):
        a = 2 * torch.rand(args.num_envs, res["action_dim"], device=u.device) - 1
        obs, rew, term, trunc, _ = env.step(a)
        finite &= bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
        for name in u.reward_manager.active_terms:
            cfg = u.reward_manager.get_term_cfg(name)
            v = cfg.func(u, **cfg.params)
            finite &= bool(torch.isfinite(v).all())
            lo_hi[name][0] = min(lo_hi[name][0], float(v.min()))
            lo_hi[name][1] = max(lo_hi[name][1], float(v.max()))
    res.update({
        "step_obs_shape": list(obs["policy"].shape), "reward_shape": list(rew.shape), "reward_dtype": str(rew.dtype),
        "terminated_dtype": str(term.dtype), "truncated_dtype": str(trunc.dtype),
        "all_finite": finite, "reward_term_ranges": lo_hi,
        "out_of_range": [n for n, (lo, hi) in lo_hi.items() if n in RANGES and (lo < RANGES[n][0] - 1e-6 or hi > RANGES[n][1] + 1e-6)],
        "unknown_terms": [n for n in lo_hi if n not in RANGES],
        "nan_guard_events": u.nan_guard_events,
    })
    print("CHECK_ENV=" + json.dumps(res))
    env.close()
