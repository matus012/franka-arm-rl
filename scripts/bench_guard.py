"""A/B timing of the NaN-guard variants in one process (same env, same GPU state).

Variants: none (no per-physics-step work), scrub (sync-free, current), sync (run-1: detect + host sync after
every physics step). Each: `--steps` control steps of random actions on `--num_envs` envs.
Usage: uv run python scripts/bench_guard.py --task LiftRL-Stock-Newton-Fix2-v0 --num_envs 4096 --steps 120
"""

from __future__ import annotations

import argparse
import json
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
p.add_argument("--task", default="LiftRL-Stock-Newton-Fix2-v0")
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--steps", type=int, default=120)
args = p.parse_args()

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.sim.device = "cuda:0"
assert_run_preconditions(cfg)

with launch_simulation(cfg, {"headless": True}):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    env.reset(seed=0)
    real_scrub = u._scrub_physics_step

    def sync_variant():
        bad = u._detect_nonfinite()
        if bool(bad.any()):
            u._nan_flag |= bad

    variants = {"none": lambda: None, "scrub": real_scrub, "sync": sync_variant}
    res = {}
    for rep in range(2):  # two passes, report the second (warm)
        for name, fn in variants.items():
            u._scrub_physics_step = fn
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args.steps):
                a = 0.3 * (2 * torch.rand(u.num_envs, u.action_manager.total_action_dim, device=u.device) - 1)
                env.step(a)
            torch.cuda.synchronize()
            res[name] = round((time.time() - t0) / args.steps * 1000, 2)
    print("BENCH_MS_PER_CONTROL_STEP=" + json.dumps(res))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
