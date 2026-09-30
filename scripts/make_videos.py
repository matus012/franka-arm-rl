"""Replay a checkpoint and write the demo videos (cameras exist only here, never in training).

  uv run python scripts/make_videos.py --task LiftRL-Own-v0 --checkpoint <model.pt> --out videos/m1
      -> grid_3x3.mp4 (+ grid_3x3.gif), single_success.mp4, single_near_miss.mp4, single_failure.mp4, videos.json
  uv run python scripts/make_videos.py --task LiftRL-Own-v0 --checkpoint <A.pt> --compare <B.pt> \
      --labels baseline twist --out videos/compare
      -> side_by_side.mp4: the same 4 episodes (same eval seed) under policy A (left) and B (right)

Episodes are classified with the evaluator's detectors: success (3 cm), near miss (not success but
final distance < 10 cm), failure (everything else). A class with no episode is reported as missing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--compare", default=None, help="second checkpoint for the side-by-side video")
p.add_argument("--labels", nargs=2, default=["baseline", "twist"])
p.add_argument("--compare_task", default=None, help="task of the --compare checkpoint (default: --task)")
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=16)
p.add_argument("--seed", type=int, default=2024)
p.add_argument("--width", type=int, default=320)
p.add_argument("--height", type=int, default=240)
p.add_argument("--closeup", type=int, default=0,
               help="close-up mode (brief 10.3): record this many envs at 1280x720, ~0.6 m from the grasp, "
                    "first 2 s at 0.25x; writes closeup_env<i>.mp4 + contact sheets to results/audit/")
p.add_argument("--tag", default="", help="prefix for close-up audit sheets")
p.add_argument("--frames_npz", default=None, help=argparse.SUPPRESS)  # internal: dump raw frames for compare
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

if args.compare and args.frames_npz is None:
    # run each policy in its own process (one sim per process), same seed -> same episodes, then stitch
    import imageio
    import numpy as np

    from lift_rl.camera import SlowMoWriter

    dumps = []
    for i, (ck, tk) in enumerate(((args.checkpoint, args.task), (args.compare, args.compare_task or args.task))):
        f = out / f"_frames_{i}.npz"
        cmd = [sys.executable, "-u", __file__, "--task", tk, "--checkpoint", ck, "--out", str(out),
               "--num_envs", "4", "--seed", str(args.seed), "--width", str(args.width), "--height", str(args.height),
               "--frames_npz", str(f)] + (["--closeup", "4"] if args.closeup else [])
        subprocess.run(cmd, check=True)
        with np.load(f) as z:  # load into memory: an open .npz cannot be deleted on Windows
            dumps.append({k: z[k] for k in z.files})
    a, b = dumps[0]["grid"], dumps[1]["grid"]
    n = min(len(a), len(b))
    bar = np.full((n, a.shape[1], 8, 3), 255, np.uint8)
    both = np.concatenate([a[:n], bar[:n], b[:n]], axis=2)
    if args.closeup:  # per-step frames: first 2 s at 0.25x, then real time (SlowMoWriter)
        w = SlowMoWriter(out / "side_by_side.mp4")
        for fr in both:
            w.add(fr)
        w.close()
    else:
        imageio.mimsave(out / "side_by_side.mp4", list(both), fps=25)
    meta = {"left": {"label": args.labels[0], "task": args.task, "checkpoint": args.checkpoint,
                     "outcomes": dumps[0]["outcomes"].tolist()},
            "right": {"label": args.labels[1], "task": args.compare_task or args.task, "checkpoint": args.compare,
                      "outcomes": dumps[1]["outcomes"].tolist()},
            "seed": args.seed, "episodes": "envs 0-3 as a 2x2 grid"}
    (out / "side_by_side.json").write_text(json.dumps(meta, indent=2))
    for i in range(2):
        (out / f"_frames_{i}.npz").unlink()
    print(json.dumps(meta, indent=2))
    sys.exit(0)

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.camera import SlowMoWriter, add_closeup_camera, add_video_camera, aim_closeup, aim_video_camera, grab_rgb, tile  # noqa: E402
from lift_rl.evaluation import make_eval_cfg, rollout  # noqa: E402
from lift_rl.physics import assert_run_preconditions  # noqa: E402
from lift_rl.policy import load_policy  # noqa: E402

if args.closeup:
    args.num_envs = args.closeup
env_cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), args.num_envs, args.seed)
if args.closeup:
    add_closeup_camera(env_cfg)
else:
    add_video_camera(env_cfg, args.width, args.height)
assert_run_preconditions(env_cfg)

SHEET_STEPS = (0, 8, 16, 25, 35, 50, 70, 100, 130, 170, 210, 249)

with launch_simulation(env_cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(args.task, cfg=env_cfg)
    wrapped, policy, _ = load_policy(env, args.task, args.checkpoint)
    N = env.unwrapped.num_envs
    if args.closeup:
        writers = [SlowMoWriter(out / f"closeup_env{i}.mp4", keep_steps=SHEET_STEPS) for i in range(N)]
        grid_w = SlowMoWriter(out / "closeup_grid_3x3.mp4") if N >= 9 and not args.frames_npz else None
        gif_frames = []
        cmp_frames = []  # compare-mode worker: 2x2 close-up tiles, one per 50 Hz step

        def on_step_c(t, u):
            if t == 0:
                aim_closeup(env)  # after the rollout's reset: aim at each env's cube start position
            fr = grab_rgb(env)
            for i, w in enumerate(writers):
                w.add(fr[i])
            if args.frames_npz:
                cmp_frames.append(tile(fr[:4, ::2, ::2], 2))
            if grid_w is not None:
                g = tile(fr[:9, ::2, ::2], 3)  # 9 x (640x360) -> 1920x1080
                grid_w.add(g)
                if t % 4 == 0:
                    gif_frames.append(g[::4, ::4])

        res = rollout(wrapped, policy, on_step=on_step_c)
        if args.frames_npz:
            succ_w = res["success"].cpu().numpy()
            dist_w = res["dist"].cpu().numpy()
            oc = np.where(succ_w, "success", np.where((~succ_w) & (dist_w < 0.10), "near_miss", "failure"))
            clean_w = res["clean"].cpu().numpy()
            np.savez_compressed(args.frames_npz, grid=np.stack(cmp_frames), outcomes=np.array(
                [f"{o}{' clean' if c else ''}" for o, c in zip(oc[:4], clean_w[:4])]))
            for w in writers:
                w.close()
            env.close()
            sys.exit(0)
        if grid_w is not None:
            grid_w.close()
            imageio.mimsave(out / "closeup_grid_3x3.gif", gif_frames, duration=80, loop=0)
        audit = ROOT / "results" / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        per_env = []
        for i, w in enumerate(writers):
            w.close()
            sheet = tile(np.stack([w.kept[k] for k in SHEET_STEPS]), 4)
            name = f"{args.tag}closeup_env{i}_steps_{'-'.join(map(str, SHEET_STEPS))}.png"
            imageio.imwrite(audit / name, sheet[::2, ::2])
            per_env.append({"env": i, "sheet": name, **{k: (bool(res[k][i]) if res[k].dtype == torch.bool else round(float(res[k][i]), 3))
                            for k in ("clean", "clean_c1_no_table_hit", "clean_c2_top_down", "clean_c3_both_pads_at_liftoff",
                                      "clean_c4_no_palm", "success", "lift", "slam", "jam", "dist", "max_robot_table_force",
                                      "max_descent_tilt_deg", "push_before_liftoff_m")}})
        import shutil

        succ_c = res["success"].cpu().numpy()
        dist_c = res["dist"].cpu().numpy()
        outcome_c = np.where(succ_c, "success", np.where((~succ_c) & (dist_c < 0.10), "near_miss", "failure"))
        singles = {}
        for cls in ("success", "near_miss", "failure"):
            idx = np.flatnonzero(outcome_c == cls)
            singles[cls] = None
            if len(idx):
                i = int(idx[0])
                shutil.copyfile(out / f"closeup_env{i}.mp4", out / f"single_{cls}_closeup.mp4")
                singles[cls] = {"env": i, "final_dist_m": float(dist_c[i]), "clean": bool(res["clean"][i])}
        meta = {"task": args.task, "checkpoint": args.checkpoint, "seed": args.seed, "mode": "closeup",
                "grid_envs": list(range(min(9, N))), "grid_outcomes": outcome_c[:9].tolist(), "singles": singles,
                "per_env": per_env}
        (out / "closeup.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(per_env, indent=1))
        print(f"NAN_GUARD_EVENTS={env.unwrapped.nan_guard_events}")
        env.close()
        sys.exit(0)
    aim_video_camera(env)
    frames: list[np.ndarray] = []  # (N,H,W,3) every 2nd control step -> 25 fps real time

    def on_step(t, u):
        if t % 2 == 0:
            frames.append(grab_rgb(env))

    res = rollout(wrapped, policy, on_step=on_step)
    video = np.stack(frames, axis=1)  # (N, T, H, W, 3)
    succ = res["success"].cpu().numpy()
    dist = res["dist"].cpu().numpy()
    near = (~succ) & (dist < 0.10)
    outcome = np.where(succ, "success", np.where(near, "near_miss", "failure"))

    if args.frames_npz:  # compare mode worker
        grid = np.stack([tile(video[:4, t], 2) for t in range(video.shape[1])])
        np.savez_compressed(args.frames_npz, grid=grid, outcomes=outcome[:4])
        env.close()
        sys.exit(0)

    grid = [tile(video[:9, t], 3) for t in range(video.shape[1])]
    imageio.mimsave(out / "grid_3x3.mp4", grid, fps=25)
    small = [g[::2, ::2] for g in grid[::2]]
    imageio.mimsave(out / "grid_3x3.gif", small, duration=80, loop=0)
    picks = {}
    for cls in ("success", "near_miss", "failure"):
        idx = np.flatnonzero(outcome == cls)
        if len(idx):
            i = int(idx[0])
            imageio.mimsave(out / f"single_{cls}.mp4", list(video[i]), fps=25)
            picks[cls] = {"env": i, "final_dist_m": float(dist[i]), "slam": bool(res["slam"][i]), "jam": bool(res["jam"][i]),
                          "clean": bool(res["clean"][i])}
        else:
            picks[cls] = None
    meta = {"task": args.task, "checkpoint": args.checkpoint, "seed": args.seed, "num_envs": N,
            "grid_envs": list(range(9)), "grid_outcomes": outcome[:9].tolist(), "singles": picks,
            "per_env": [{"env": i, "outcome": str(outcome[i]), "final_dist_m": float(dist[i]),
                         "slam": bool(res["slam"][i]), "jam": bool(res["jam"][i]), "clean": bool(res["clean"][i]),
                         "max_table_force_N": float(res["max_table_force"][i]), "max_jam_run": int(res["max_jam_run"][i])}
                        for i in range(N)]}
    (out / "videos.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("grid_outcomes", "singles")}, indent=2))
    print(f"NAN_GUARD_EVENTS={env.unwrapped.nan_guard_events}")
    env.close()
