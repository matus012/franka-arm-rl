"""Diagnostic (eval-only): roll out a checkpoint on the staged env and report, at chosen steps, where the hand
parks relative to the staged-reward conditions (descent progress h, gripper command, xy/tilt/yaw error, stage).

Usage: uv run python scripts/diag_policy_state.py --checkpoint <model.pt> --num_envs 200 --seed 1001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.evaluation import make_eval_cfg
from lift_rl.physics import assert_run_preconditions
from lift_rl.policy import load_policy
from lift_rl.staged import CUBE_HALF, TABLE_TOP_Z

p = argparse.ArgumentParser()
p.add_argument("--task", default="LiftRL-Staged-v0")
p.add_argument("--checkpoint", required=True)
p.add_argument("--num_envs", type=int, default=200)
p.add_argument("--seed", type=int, default=1001)
p.add_argument("--out", default="")
args = p.parse_args()

cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), args.num_envs, args.seed)
assert_run_preconditions(cfg)
STEPS = (50, 100, 200, 249)

with launch_simulation(cfg, {"headless": True}):
    env = gym.make(args.task, cfg=cfg)
    wrapped, policy, _ = load_policy(env, args.task, args.checkpoint)
    u = env.unwrapped
    obs = wrapped.reset()[0]
    rows = {}
    q = lambda t, x: [round(float(v), 4) for v in torch.quantile(t.float(), torch.tensor(x, device=t.device))]  # noqa: E731
    with torch.inference_mode():
        for t in range(250):
            obs = wrapped.step(policy(obs))[0]
            if t in STEPS:
                st = u._staged_state.get()
                ee = st["ee"]
                cube = u.scene["object"].data.root_pos_w.torch - u.scene.env_origins
                pre_z = cube[:, 2] + CUBE_HALF + 0.10
                z_grasp = torch.clamp(cube[:, 2], min=TABLE_TOP_Z + 0.01)
                h = ((pre_z - ee[:, 2]) / (pre_z - z_grasp)).clamp(0, 1)
                grip = u.action_manager.get_term("gripper_action").raw_actions[:, 0]
                xy = torch.linalg.norm(ee[:, :2] - cube[:, :2], dim=-1)
                rows[t] = {
                    "stage_counts": [int((st["stage"] == k).sum()) for k in range(5)],
                    "descended_frac": float(st["descended"].float().mean()),
                    "h_q10_50_90": q(h, [0.1, 0.5, 0.9]),
                    "frac_h_ge_0.9": float((h >= 0.9).float().mean()),
                    "ee_minus_cube_center_z_q10_50_90_m": q(ee[:, 2] - cube[:, 2], [0.1, 0.5, 0.9]),
                    "grip_raw_action_q10_50_90": q(grip, [0.1, 0.5, 0.9]),
                    "frac_grip_closed_cmd": float((grip < 0).float().mean()),
                    "xy_err_q50_m": q(xy, [0.5])[0],
                    "speed_q50": q(st["speed"], [0.5])[0],
                }
    print("DIAG=" + json.dumps(rows))
    if args.out:
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "seed": args.seed, "steps": rows}, indent=2))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
