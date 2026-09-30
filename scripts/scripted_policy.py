"""Instrument proof for the clean-grasp detector (brief 10.3 step 1): hand-coded policies on fix-2 physics,
classified by the evaluator's own EpisodeTracker.

  --mode clean      top-down: hover 12 cm above the cube, slow ramp down to grasp height, close, slow lift
  --mode rake_hard  the run-1 exploit: hand tilted 40 deg, lunge past the cube and into the table, drag the
                    cube back toward the robot, close, lift
  --mode rake_soft  same lunge-and-drag, tilted, but 1.2 cm above the table (no table hit): tests that the
                    top-down criterion catches a rake on its own

IK-abs action variant of the stock scene (same cube, gripper and physics as training). Writes
results/scripted_<mode>.json, close-up mp4s of envs 0-1 and a close-up contact sheet (results/audit/).
Usage: uv run python scripts/scripted_policy.py --mode clean --num_envs 64
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
import imageio
import numpy as np
import torch

import lift_rl  # noqa: F401
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.camera import SlowMoWriter, add_closeup_camera, aim_closeup, grab_rgb, tile
from lift_rl.evaluation import contact_readings, cube_and_target, hand_pose
from lift_rl.instrument import add_eval_sensors
from lift_rl.metrics import EpisodeTracker, Thresholds, summarize
from lift_rl.physics import assert_run_preconditions, newton_mjwarp_cfg
from lift_rl.stock_cfg import PHYSICS_FIX2

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("clean", "rake_hard", "rake_soft"), required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--video_envs", type=int, default=2)
p.add_argument("--no_video", action="store_true")
p.add_argument("--debug_env", type=int, default=-1, help="print a per-step contact/pose trace for this env")
args = p.parse_args()

TASK = "LiftRL-Stock-IK-Newton-v0"
cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.sim.device = "cuda:0"
cfg.episode_length_s = 10.0
cfg.terminations.object_dropping = None
f = PHYSICS_FIX2
cfg.sim.physics = newton_mjwarp_cfg(num_substeps=f["num_substeps"], contact_ke=f["contact_ke"], contact_kd=f["contact_kd"])
cfg.scene.robot.actuators["panda_hand"].armature = f["finger_armature"]
add_eval_sensors(cfg)
video = not args.no_video
if video:
    add_closeup_camera(cfg)
assert_run_preconditions(cfg)
KEEP = tuple(range(0, 250, 10))

with launch_simulation(cfg, {"headless": True, "enable_cameras": video}):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    env.reset()
    robot, obj = u.scene["robot"], u.scene["object"]
    arm = u.action_manager.get_term("arm_action")
    N, dev = u.num_envs, u.device
    ep, eq = arm._compute_frame_pose()
    hold = torch.cat([ep, eq, torch.ones(N, 1, device=dev)], 1)
    for _ in range(10):  # settle: hold the start pose, gripper open
        env.step(hold)
    c0 = (obj.data.root_pos_w.torch - robot.data.root_pos_w.torch).clone()
    q_down = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(N, 1)  # xyzw, hand z -> -z world
    tilt = torch.tensor(math.radians(-40.0), device=dev)
    q_tilt = quat_mul(quat_from_euler_xyz(torch.zeros(N, device=dev), tilt.repeat(N), torch.zeros(N, device=dev)), q_down)
    if video:
        aim_closeup(env)
        (ROOT / "results" / "videos_scripted").mkdir(parents=True, exist_ok=True)
        writers = [SlowMoWriter(ROOT / "results" / "videos_scripted" / f"{args.mode}_env{i}.mp4", keep_steps=KEEP)
                   for i in range(min(args.video_envs, N))]
    th = Thresholds()
    tracker = EpisodeTracker(N, dev, th)
    from lift_rl.staged import StagedParams
    from lift_rl.staged_cfg import StagedState

    staged = StagedState(u, StagedParams())  # the training reward, evaluated on the scripted trajectory
    max_stage = torch.zeros(N, dtype=torch.long, device=dev)
    staged_return = torch.zeros(N, device=dev)
    integ = torch.zeros(N, 3, device=dev)
    V = lambda x, y, z: torch.tensor([x, y, z], device=dev)  # noqa: E731

    def plan(t: int):
        """(target position in robot base frame, orientation, gripper, integrator freeze mode)."""
        if args.mode == "clean":
            # fingertip center 6 mm above the cube center; closed-loop (rate-limited integrator) so the hand
            # hull (its flat bottom is 6.6 cm below the hand origin) stays clear of the cube top
            g_z = 0.006
            if t < 40:  # first go high: the swing from the tilted start pose otherwise sweeps low over the cube
                return c0 + V(0, 0, 0.25), q_down, 1.0, "none"
            if t < 90:  # hover until xy and the IK offset have converged
                return c0 + V(0, 0, 0.12), q_down, 1.0, "none"
            if t < 150:  # slow vertical descent (~0.1 m/s), z integrator frozen (no windup overshoot)
                s = (t - 90) / 60.0
                return c0 + V(0, 0, 0.12 * (1 - s) + g_z * s), q_down, 1.0, "none"
            if t < 170:
                return c0 + V(0, 0, g_z), q_down, 1.0, "none"
            if t < 195:
                return c0 + V(0, 0, g_z), q_down, -1.0, "all"
            s = min((t - 195) / 45.0, 1.0)
            return c0 + V(0, 0, g_z + 0.15 * s), q_down, -1.0, "none"
        z_low = -0.015 if args.mode == "rake_hard" else 0.012
        if t < 50:
            return c0 + V(0.14, 0, 0.12), q_tilt, 1.0, "none"
        if t < 80:
            s = (t - 50) / 30.0
            return c0 + V(0.14 - 0.04 * s, 0, 0.12 * (1 - s) + z_low * s), q_tilt, 1.0, "none"
        if t < 140:
            s = (t - 80) / 60.0
            return c0 + V(0.10 - 0.13 * s, 0, z_low), q_tilt, 1.0, "none"
        if t < 165:
            return c0 + V(-0.03, 0, z_low), q_tilt, -1.0, "all"
        s = min((t - 165) / 50.0, 1.0)
        return c0 + V(-0.03, 0, z_low + 0.20 * s), q_tilt, -1.0, "none"

    for t in range(250):
        tgt, q, g, freeze = plan(t)  # freeze: "none" | "z" (xy integrator only) | "all"
        e, _ = arm._compute_frame_pose()
        if freeze != "all":
            step = (0.1 * (tgt - e)).clamp(-0.002, 0.002)  # rate-limited: no windup overshoot
            if freeze == "z":
                step[:, 2] = 0.0
            integ = (integ + step).clamp(-0.15, 0.15)
        env.step(torch.cat([tgt + integ, q, torch.full((N, 1), g, device=dev)], 1))
        c = contact_readings(u)
        st = staged.get()
        max_stage = torch.maximum(max_stage, st["stage"])
        staged_return += st["reward"] * u.step_dt
        cube, _t = cube_and_target(u)
        ee_z, tilt_deg = hand_pose(u)
        tracker.update(c["hand_table"], cube[:, 2], c["palm"], c["lf"], c["rf"], u._nan_flag, arm_force=c["arm"],
                       robot_table_force=c["robot_table"], ee_z=ee_z, hand_tilt_deg=tilt_deg, cube_xy=cube[:, :2])
        if args.debug_env >= 0 and (t % 5 == 0 or c["palm"][args.debug_env] > 1.0):
            i = args.debug_env
            hz = robot.data.body_pos_w.torch[i, robot.body_names.index("panda_hand"), 2] - u.scene.env_origins[i, 2]
            print(f"[dbg] t={t} ee_z={ee_z[i]:.4f} hand_z={hz:.4f} cube_z={cube[i, 2]:.4f} tilt={tilt_deg[i]:.1f} "
                  f"palm={c['palm'][i]:.1f} lf={c['lf'][i]:.1f} rf={c['rf'][i]:.1f} arm={c['arm'][i]:.1f} "
                  f"table={c['robot_table'][i]:.1f} grip={robot.data.joint_pos.torch[i, -2:].sum():.4f}")
        if video:
            frames = grab_rgb(env)
            for i, w in enumerate(writers):
                w.add(frames[i])
    cube, target = cube_and_target(u)
    out = tracker.final(cube, target, contact_readings(u)["cube_table"])
    s = summarize(out)
    res = {"mode": args.mode, "num_envs": N, "seed": args.seed, "physics": "fix2", **s,
           "max_robot_table_force_N_quantiles": [float(torch.quantile(out["max_robot_table_force"], q)) for q in (0.5, 0.9)],
           "max_descent_tilt_deg_median": float(out["max_descent_tilt_deg"].median()),
           "nan_guard_events": u.nan_guard_events,
           "staged_reward": {"max_stage_reached_counts": [int((max_stage == k).sum()) for k in range(5)],
                             "episode_return_mean": float(staged_return.mean()),
                             "episode_return_quartiles": [float(torch.quantile(staged_return, q)) for q in (0.25, 0.5, 0.75)]},
           "per_env_first8": [{k: (bool(out[k][i]) if out[k].dtype == torch.bool else round(float(out[k][i]), 3))
                               for k in ("clean", "clean_c1_no_table_hit", "clean_c2_top_down",
                                         "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm", "lift",
                                         "max_robot_table_force", "max_descent_tilt_deg", "push_before_liftoff_m")}
                              for i in range(min(8, N))]}
    print("STAGED=" + json.dumps(res["staged_reward"]))
    print("SCRIPTED=" + json.dumps({k: res[k] for k in ("mode", "clean", "clean_c1_no_table_hit", "clean_c2_top_down",
                                                         "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm", "lift")}))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"scripted_{args.mode}.json").write_text(json.dumps(res, indent=2))
    if video:
        audit = ROOT / "results" / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(writers):
            w.close()
            steps = [10, 50, 70, 90, 120, 150, 170, 220] if args.mode != "clean" else [10, 60, 90, 110, 130, 150, 180, 220]
            sheet = tile(np.stack([w.kept[k] for k in steps]), 4)
            imageio.imwrite(audit / f"scripted_{args.mode}_env{i}_steps_{'-'.join(map(str, steps))}.png", sheet[::2, ::2])
    env.close()
