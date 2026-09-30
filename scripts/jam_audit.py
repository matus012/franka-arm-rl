"""Jam/slam-detector audit (--mode jam|slam): render N episodes, pick up to K jam-flagged and K clean episodes, and write one
contact sheet (PNG, 8 close-up frames) per episode so a human can check the detector by eye.

  uv run python scripts/jam_audit.py --task LiftRL-Stock-Newton-v0 --checkpoint <model.pt> --out logs/jam_audit/m0

Clean candidates are taken from episodes where the robot touched the cube (otherwise "clean" is trivial).
Frames are centred on the moment of the longest jam run (flagged) or of the most cube contact (clean).
Writes <out>/audit.json with per-episode detector evidence; the by-eye verdicts were recorded in the development log.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import imageio
import numpy as np
import torch

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.camera import add_video_camera, grab_rgb, tile
from lift_rl.evaluation import contact_readings, cube_and_target, make_eval_cfg, rollout
from lift_rl.metrics import EpisodeTracker, Thresholds
from lift_rl.physics import assert_run_preconditions
from lift_rl.policy import load_policy

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=48)
p.add_argument("--k", type=int, default=10)
p.add_argument("--seed", type=int, default=777)
p.add_argument("--mode", choices=("jam", "slam"), default="jam")
args = p.parse_args()

cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), args.num_envs, args.seed)
add_video_camera(cfg, 200, 150)
assert_run_preconditions(cfg)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
th = Thresholds()

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(args.task, cfg=cfg)
    wrapped, policy, _ = load_policy(env, args.task, args.checkpoint)
    u = env.unwrapped
    N = u.num_envs
    helper = EpisodeTracker(N, u.device, th)
    frames, jam_steps, touch, table_f, table_body = [], [], [], [], []
    ht_sensor = u.scene["contact_hand_table"]

    def on_step(t, uu):
        c = contact_readings(uu)
        cube, _ = cube_and_target(uu)
        jam_steps.append(helper.jam_step(cube[:, 2], c["palm"], c["lf"], c["rf"], c["arm"]).cpu())
        touch.append(((c["palm"] > th.contact_n) | (c["lf"] > th.contact_n) | (c["rf"] > th.contact_n) | (c["arm"] > th.contact_n)).cpu())
        fm = ht_sensor.data.force_matrix_w
        fm = (fm.torch if hasattr(fm, "torch") else fm).reshape(uu.num_envs, -1, 3).norm(dim=-1)  # (N, bodies)
        table_f.append(fm.amax(1).cpu())
        table_body.append(fm.argmax(1).cpu())
        # close-up camera follows the cube (update_latest_camera_pose=True)
        cam = uu.scene["video_cam"]
        cp = uu.scene["object"].data.root_pos_w
        cp = cp.torch if hasattr(cp, "torch") else cp
        cam.set_world_poses_from_view(cp + torch.tensor([0.45, 0.30, 0.25], device=uu.device), cp)
        if t % 2 == 0:
            frames.append(grab_rgb(env))

    res = rollout(wrapped, policy, th, on_step=on_step)
    J = torch.stack(jam_steps, 1).numpy()  # (N, T)
    Tc = torch.stack(touch, 1).numpy()
    video = np.stack(frames, 1)  # (N, T/2, H, W, 3)
    F = torch.stack(table_f, 1).numpy()  # (N, T) max hand/finger-table force per step
    FB = torch.stack(table_body, 1).numpy()
    body_names = list(ht_sensor.sensor_names)
    if args.mode == "jam":
        flag = res["jam"].cpu().numpy()
        clean_pool = [i for i in np.flatnonzero(~flag) if Tc[i].sum() >= 10]
    else:
        flag = res["slam"].cpu().numpy()
        clean_pool = list(np.flatnonzero(~flag))
    flagged = list(np.flatnonzero(flag))[: args.k]
    clean = clean_pool[: args.k]
    jam = flag
    rows = []
    for cls, ids in ((args.mode, flagged), ("clean", clean)):
        for i in ids:
            if cls == "slam":  # center on the peak table force
                center = int(np.argmax(F[i]))
            elif cls == "jam":
                runs, best, cur, end = J[i], 0, 0, 0
                for t, v in enumerate(runs):
                    cur = cur + 1 if v else 0
                    if cur > best:
                        best, end = cur, t
                center = end - best // 2
            else:
                center = int(np.argmax(np.convolve(Tc[i].astype(float), np.ones(20), "same")))
            W = 14 if args.mode == "slam" else 40
            ts = np.clip(np.linspace(center - W, center + W, 8).astype(int), 0, J.shape[1] - 1)
            sheet = tile(np.stack([video[i, t // 2] for t in ts]), 4)
            name = f"{cls}_env{i:02d}.png"
            imageio.imwrite(out / name, sheet)
            rows.append({"file": name, "class": cls, "env": int(i), "frame_steps": ts.tolist(),
                         "jam_steps_total": int(J[i].sum()), "max_jam_run": int(res["max_jam_run"][i]),
                         "touch_steps": int(Tc[i].sum()), "final_dist_m": float(res["dist"][i]),
                         "lift": bool(res["lift"][i]), "success": bool(res["success"][i]),
                         "peak_table_force_N": float(F[i].max()), "peak_step": int(np.argmax(F[i])),
                         "peak_body": body_names[int(FB[i, int(np.argmax(F[i]))])] if body_names else None,
                         "steps_over_20N": int((F[i] > 20).sum()),
                         "force_trace_every5": [round(float(x), 1) for x in F[i][::5]]})
    (out / "audit.json").write_text(json.dumps({"task": args.task, "checkpoint": args.checkpoint, "seed": args.seed,
                                                "num_envs": N, "mode": args.mode, "flagged_total": int(jam.sum()),
                                                "episodes": rows}, indent=2))
    print(f"flagged {len(flagged)} / clean {len(clean)} (jam total {int(jam.sum())} of {N})")
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
