"""Build an env, print scene layout (bodies, shapes), step random actions, report speed.

Usage: uv run python scripts/smoke_env.py --task LiftRL-Stock-Newton-v0 --num_envs 64 --steps 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import torch

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.physics import assert_run_preconditions

p = argparse.ArgumentParser()
p.add_argument("--task", default="LiftRL-Stock-Newton-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=100)
p.add_argument("--layout", action="store_true")
args = p.parse_args()

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = "cuda:0"
assert_run_preconditions(env_cfg)

with launch_simulation(env_cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    print(f"[smoke] env built in {time.time() - t0:.1f}s, device={u.device}")
    obs, _ = env.reset()
    print("[smoke] obs:", {k: (tuple(v.shape), v.dtype) for k, v in obs.items()})
    print("[smoke] action dim:", u.action_manager.total_action_dim)
    if args.layout:
        from isaaclab_newton.physics import NewtonManager as M

        m = M._model
        labels = getattr(m, "body_label", None) or getattr(m, "body_key", None)
        print("[smoke] bodies env0:", [b for b in labels if "env_0/" in b])
        sl = getattr(m, "shape_label", None) or getattr(m, "shape_key", None)
        print("[smoke] shapes env0:", [s for s in sl if "env_0/" in s])
        print("[smoke] world_count", m.world_count, "joint_coord_count", m.joint_coord_count, "body_count", m.body_count)
        print("[smoke] robot bodies:", u.scene["robot"].data.body_names)
        print("[smoke] object root z:", u.scene["object"].data.root_pos_w.torch[:4, 2].tolist())
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(args.steps):
        a = 2 * torch.rand(args.num_envs, u.action_manager.total_action_dim, device=u.device) - 1
        obs, r, te, tr, ex = env.step(a)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[smoke] {args.steps} steps in {dt:.2f}s -> {args.steps * args.num_envs / dt:.0f} env-steps/s")
    print(f"[smoke] reward finite: {torch.isfinite(r).all().item()}, mean {r.mean().item():.4f}")
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    print(f"[smoke] peak cuda mem (torch): {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    env.close()
