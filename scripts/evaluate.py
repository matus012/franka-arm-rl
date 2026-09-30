"""Evaluate a checkpoint: 1,000 episodes, fixed eval seed, deterministic actions, section-6 metrics.

Usage:
  uv run python scripts/evaluate.py --task LiftRL-Stock-Newton-v0 --checkpoint logs/rsl_rl/.../model_1499.pt \
      --out results/m0_eval.json [--num_envs 1000] [--seed 12345] [--cube_mass 0.216] [--cube_friction 1.0]
Writes the JSON summary (commit it) and a per-episode .pt next to it under logs/eval/ (not committed).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.evaluation import make_eval_cfg, rollout
from lift_rl.metrics import Thresholds, summarize
from lift_rl.physics import assert_on_cuda, assert_run_preconditions
from lift_rl.policy import load_policy
from lift_rl.randomization import set_fixed_cube_physics

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=1000)
p.add_argument("--num_episodes", type=int, default=1000)
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--cube_mass", type=float, default=None, help="robustness grid: fixed cube mass [kg]")
p.add_argument("--cube_friction", type=float, default=None, help="robustness grid: fixed cube friction")
args = p.parse_args()

env_cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), args.num_envs, args.seed)
if args.cube_mass is not None or args.cube_friction is not None:
    set_fixed_cube_physics(env_cfg, args.cube_mass, args.cube_friction)
assert_run_preconditions(env_cfg)
th = Thresholds()

with launch_simulation(env_cfg, {"headless": True}):
    t0 = time.time()
    env = gym.make(args.task, cfg=env_cfg)
    wrapped, policy, policy_module = load_policy(env, args.task, args.checkpoint)
    assert_on_cuda(env, policy_module)
    # proof the robustness-grid settings took effect: mass and friction as the simulator sees them
    import warp as wp
    from isaaclab_newton.physics import NewtonManager

    _obj = env.unwrapped.scene["object"]
    _m = _obj.data.body_mass
    applied = {"cube_mass_kg_mean": float((_m.torch if hasattr(_m, "torch") else _m).mean())}
    try:
        _mu = wp.to_torch(_obj._root_view.get_attribute("shape_material_mu", NewtonManager.get_model())[:, 0])
        applied["cube_mu_mean"] = float(_mu.mean())
    except Exception as err:  # noqa: BLE001
        applied["cube_mu_mean"] = f"unavailable: {err}"
    batches = []
    while sum(b["success"].numel() for b in batches) < args.num_episodes:
        batches.append(rollout(wrapped, policy, th))
    outcomes = {k: torch.cat([b[k] for b in batches])[: args.num_episodes] for k in batches[0]}
    summary = summarize(outcomes)
    u = env.unwrapped
    res = {
        "task": args.task,
        "checkpoint": str(Path(args.checkpoint).resolve().relative_to(ROOT)) if Path(args.checkpoint).resolve().is_relative_to(ROOT) else args.checkpoint,
        "eval_seed": args.seed,
        "num_envs": args.num_envs,
        "cube_mass": args.cube_mass,
        "cube_friction": args.cube_friction,
        "applied_physics": applied,
        "thresholds": th.to_dict(),
        "nan_guard_events": u.nan_guard_events,
        "wall_s": round(time.time() - t0, 1),
        **summary,
    }
    print(json.dumps(res, indent=2))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    per_ep = ROOT / "logs" / "eval" / (out.stem + "_episodes.pt")
    per_ep.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in outcomes.items()}, per_ep)
    env.close()
