"""NaN-guard check: inject NaN into one world's solver data and Newton state, step, and verify that the
guard fires once for that world only, gives it reward 0 + a terminal step, and leaves every world finite
afterwards. Prints one JSON line prefixed with CHECK_NAN=.

Usage: uv run python scripts/check_nan_guard.py --task LiftRL-Own-v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import torch
import warp as wp

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.physics import assert_run_preconditions

p = argparse.ArgumentParser()
p.add_argument("--task", default="LiftRL-Own-v0")
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--bad_env", type=int, default=3)
args = p.parse_args()

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
assert_run_preconditions(cfg)

with launch_simulation(cfg, {"headless": True}):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    env.reset(seed=0)
    zero = torch.zeros(args.num_envs, u.action_manager.total_action_dim, device=u.device)
    for _ in range(5):
        env.step(zero)
    from isaaclab_newton.physics import NewtonManager as M

    d = M._solver.mjw_data
    b = args.bad_env
    for name in ("qvel", "qacc_warmstart"):
        wp.to_torch(getattr(d, name))[b] = float("nan")
    _, rew, term, trunc, _ = env.step(zero)
    first = {"events": u.nan_guard_events, "flag": u._nan_flag.tolist(), "reward_bad": float(rew[b]),
             "terminated_bad": bool(term[b]), "terminated_others": bool(term[torch.arange(args.num_envs, device=u.device) != b].any()),
             "reward_finite": bool(torch.isfinite(rew).all())}
    finite_after = True
    for _ in range(30):
        obs, rew, *_ = env.step(zero)
        finite_after &= bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
    res = {**first, "events_after_30_steps": u.nan_guard_events, "finite_after": finite_after,
           "solver_finite_after": all(bool(torch.isfinite(wp.to_torch(getattr(d, n))).all()) for n in ("qpos", "qvel", "qacc"))}
    print("CHECK_NAN=" + json.dumps(res))
    env.close()
