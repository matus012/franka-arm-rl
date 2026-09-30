"""Diagnostic: does a contact inject energy? Sweep the hand sideways through a resting cube at a fixed
speed and measure the cube's peak speed / travel. A physical (inelastic) push leaves the cube no faster
than the hand; a much faster cube means the contact solver is flinging it.

Uses the IK-abs diagnostic env. Usage:
  uv run python scripts/push_test.py --num_envs 32 [--contact_ke 1e4 --contact_kd 200] [--finger_armature 0.1] [--substeps 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.instrument import add_eval_sensors
from lift_rl.physics import assert_run_preconditions, newton_mjwarp_cfg

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=32)
p.add_argument("--speed", type=float, default=0.5, help="hand sweep speed [m/s]")
p.add_argument("--contact_ke", type=float, default=None)
p.add_argument("--contact_kd", type=float, default=None)
p.add_argument("--finger_armature", type=float, default=None)
p.add_argument("--substeps", type=int, default=2)
p.add_argument("--iterations", type=int, default=100)
p.add_argument("--solimp_dmax", type=float, default=None, help="e.g. 0.99 (default MuJoCo/Newton 0.95)")
p.add_argument("--out", default=str(ROOT / "logs" / "debug" / "push_test.json"))
args = p.parse_args()

TASK = "LiftRL-Stock-IK-Newton-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.seed = 3
cfg.sim.device = "cuda:0"
cfg.episode_length_s = 20.0
cfg.sim.physics = newton_mjwarp_cfg(iterations=args.iterations, num_substeps=args.substeps,
                                    contact_ke=args.contact_ke, contact_kd=args.contact_kd)
if args.finger_armature is not None:
    cfg.scene.robot.actuators["panda_hand"].armature = args.finger_armature
cfg.terminations.object_dropping = None
if args.solimp_dmax is not None:
    from isaaclab.managers import EventTermCfg

    from lift_rl.physics import set_contact_solimp

    cfg.events.solimp = EventTermCfg(func=set_contact_solimp, mode="startup",
                                     params={"solimp": (args.solimp_dmax - 0.04, args.solimp_dmax, 0.001, 0.5, 2.0)})
add_eval_sensors(cfg)
assert_run_preconditions(cfg)

with launch_simulation(cfg, {"headless": True}):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    env.reset()
    robot, obj = u.scene["robot"], u.scene["object"]
    arm = u.action_manager.get_term("arm_action")
    N, dev = u.num_envs, u.device
    quat = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(N, 1)
    grip = torch.full((N, 1), -1.0, device=dev)  # closed fingers = a blunt pusher
    ep, _ = arm._compute_frame_pose()
    for _ in range(10):
        env.step(torch.cat([ep, quat, grip], 1))
    c0 = (obj.data.root_pos_w.torch - robot.data.root_pos_w.torch).clone()
    start = c0 + torch.tensor([0.0, 0.15, 0.0], device=dev)  # 15 cm to the +y side, at cube-center height
    end = c0 + torch.tensor([0.0, -0.25, 0.0], device=dev)
    integ = torch.zeros(N, 3, device=dev)
    dt = u.step_dt
    hi = robot.body_names.index("panda_hand")
    hand_v, cube_v_max, travel = [], torch.zeros(N, device=dev), None
    cube_vz_max, cube_z_max = torch.zeros(N, device=dev), torch.zeros(N, device=dev)
    prev_h = robot.data.body_pos_w.torch[:, hi].clone()
    prev_c = obj.data.root_pos_w.torch.clone()
    T_move = int(0.40 / args.speed / dt)
    for t in range(60 + T_move + 50):
        if t < 60:
            tgt = start + torch.tensor([0, 0, 0.10 * max(0.0, 1 - t / 40)], device=dev)
        else:
            s = min((t - 60) / T_move, 1.0)
            tgt = start + (end - start) * s
        e, _ = arm._compute_frame_pose()
        integ = (integ + 0.1 * (tgt - e)).clamp(-0.15, 0.15)
        env.step(torch.cat([tgt + integ, quat, grip], 1))
        h, c = robot.data.body_pos_w.torch[:, hi].clone(), obj.data.root_pos_w.torch.clone()
        if t >= 60:  # finite differences (Newton body velocities are not populated for this articulation)
            hand_v.append((h - prev_h).norm(dim=1) / dt)
            cv = (c - prev_c) / dt
            cube_v_max = torch.maximum(cube_v_max, cv.norm(dim=1))
            cube_vz_max = torch.maximum(cube_vz_max, cv[:, 2])
            cube_z_max = torch.maximum(cube_z_max, c[:, 2] - u.scene.env_origins[:, 2])
        prev_h, prev_c = h, c
    hand_v = torch.stack(hand_v, 1)
    travel = (obj.data.root_pos_w.torch - robot.data.root_pos_w.torch - c0)[:, :2].norm(dim=1)
    res = {
        "substeps": args.substeps, "iterations": args.iterations, "contact_ke": args.contact_ke,
        "contact_kd": args.contact_kd, "solimp_dmax": args.solimp_dmax, "finger_armature": args.finger_armature, "commanded_speed": args.speed,
        "hand_speed_median": float(hand_v.median()), "hand_speed_p95": float(torch.quantile(hand_v.flatten(), 0.95)),
        "cube_peak_speed_median": float(cube_v_max.median()), "cube_peak_speed_max": float(cube_v_max.max()),
        "cube_travel_median_m": float(travel.median()), "cube_travel_max_m": float(travel.max()),
        "cube_peak_upward_speed_median": float(cube_vz_max.median()), "cube_peak_upward_speed_max": float(cube_vz_max.max()),
        "cube_max_height_median_m": float(cube_z_max.median()), "cube_max_height_max_m": float(cube_z_max.max()),
        "frac_cube_launched_over_2x_hand": float((cube_v_max > 2 * torch.quantile(hand_v.flatten(), 0.95)).float().mean()),
        "ratio_cube_peak_to_hand_p95": float(cube_v_max.median() / torch.quantile(hand_v.flatten(), 0.95)),
        "nan_guard_events": u.nan_guard_events,
    }
    print("PUSH=" + json.dumps(res))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    env.close()
