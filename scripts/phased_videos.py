"""Section 16 videos + audit frames on LiftRL-Phased-Fixed-v0 (phase name from the env's phase machine, target marker,
1280x720, real time 50 fps, 8 s episodes).

  --mode closeup  N envs (default 9) close-up: closeup_env<i>.mp4, closeup_grid_3x3.mp4/.gif (N >= 9),
                  single_success_closeup.mp4, audit sheets (>= 3 frames per phase reached) for --audit_envs envs
                  -> results/audit/<tag>env<i>_phase<k>_<name>_steps_*.png, and <tag>videos.json
  --mode wide     1 env, wide camera -> single_<outcome>_wide.mp4
  --mode frames   worker: 1 env close-up -> --out mp4 (+ .json)
  --mode side_by_side  scripted (left, step-A controller + reflex) vs --checkpoint (right) -> side_by_side.mp4
Policy: --checkpoint <model.pt> or --scripted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK = "LiftRL-Phased-Fixed-v0"  # --task overrides (e.g. the task-space variant)
p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("closeup", "wide", "frames", "side_by_side"), required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--scripted", action="store_true")
p.add_argument("--num_envs", type=int, default=None)
p.add_argument("--seed", type=int, default=2024)
p.add_argument("--out", required=True, help="output folder (closeup/wide/side_by_side) or mp4 file (frames)")
p.add_argument("--tag", default="")
p.add_argument("--audit_envs", type=int, default=2)
p.add_argument("--label", default=None)
p.add_argument("--task", default=None)
p.add_argument("--cube_mass", type=float, default=None)
p.add_argument("--cube_friction", type=float, default=None)
p.add_argument("--right_checkpoint", default=None, help="side_by_side: right side (default --checkpoint; left = scripted)")
p.add_argument("--left_checkpoint", default=None, help="side_by_side: left side checkpoint instead of the scripted arm")
p.add_argument("--left_label", default=None)
p.add_argument("--right_label", default=None)
args = p.parse_args()
TASK = args.task or TASK

if args.mode == "side_by_side":
    import imageio
    import numpy as np

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts, metas = [], []
    phys = (["--cube_mass", str(args.cube_mass)] if args.cube_mass is not None else []) + (
        ["--cube_friction", str(args.cube_friction)] if args.cube_friction is not None else [])
    left = (["--checkpoint", args.left_checkpoint] if args.left_checkpoint else ["--scripted"]) + (
        ["--label", args.left_label] if args.left_label else [])
    right = ["--checkpoint", args.right_checkpoint or args.checkpoint] + (["--label", args.right_label] if args.right_label else [])
    for side, extra in (("left", left + phys), ("right", right + phys)):
        f = out / f"_sbs_{side}.mp4"
        subprocess.run([sys.executable, "-u", __file__, "--mode", "frames", "--seed", str(args.seed), "--out", str(f),
                        "--num_envs", str(args.num_envs or 1)]
                       + extra + (["--task", args.task] if args.task else []), check=True)
        parts.append(f)
        metas.append(json.loads(f.with_suffix(".json").read_text()))
    ra, rb = (imageio.get_reader(str(f)) for f in parts)
    w = imageio.get_writer(str(out / f"{args.tag}side_by_side.mp4"), fps=50, codec="libx264", quality=8, macro_block_size=8)
    for fa, fb in zip(ra, rb):
        w.append_data(np.concatenate([fa, np.full((fa.shape[0], 8, 3), 255, np.uint8), fb], axis=1))
    w.close()
    ra.close()
    rb.close()
    for f in parts:
        f.unlink()
        f.with_suffix(".json").unlink()
    meta = {"seed": args.seed, "cube_mass": args.cube_mass, "cube_friction": args.cube_friction, "left": metas[0], "right": metas[1]}
    (out / f"{args.tag}side_by_side.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    sys.exit(0)

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.camera import tile  # noqa: E402
from lift_rl.evaluation import cube_and_target, make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds  # noqa: E402
from lift_rl.phases import PHASES  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from lift_rl.video import aim_task, aim_wide, cam_cfg, grab, overlay, project, target_points, writer  # noqa: E402

N = args.num_envs or {"closeup": 9, "wide": 1, "frames": 1}[args.mode]
cfg = make_eval_cfg(load_cfg_from_registry(TASK, "env_cfg_entry_point"), N, args.seed)
cfg.scene.video_cam = cam_cfg(15.0 if args.mode == "wide" else 18.0)
if args.cube_mass is not None or args.cube_friction is not None:
    from lift_rl.randomization import set_fixed_cube_physics

    set_fixed_cube_physics(cfg, args.cube_mass, args.cube_friction)
assert_run_preconditions(cfg)
th = Thresholds()
label = args.label or ("scripted (step A + reflex)" if args.scripted else f"RL: {Path(args.checkpoint).parent.name[-14:]}/{Path(args.checkpoint).stem}")

with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    if args.scripted:
        from lift_rl.scripted import ScriptedController

        ctrl = ScriptedController(env, th, reflex=True)
        run_env, policy = env, ctrl
        if "-TS-" in TASK:  # the scripted arm through the 4-D task-space interface
            from lift_rl.task_space import scripted_delta_policy

            policy = scripted_delta_policy(ctrl, u.action_manager.get_term("arm_action"))
    else:
        from lift_rl.policy import load_policy

        run_env, policy, module = load_policy(env, TASK, args.checkpoint)
        assert_on_cuda(env, module)
    cam = u.scene["video_cam"]
    steps = cfg.eval_steps
    out = Path(args.out)
    if args.mode != "frames":
        out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp" if args.mode != "frames" else None
    if tmp is not None:
        tmp.mkdir(exist_ok=True)
    ws = [writer(tmp / f"env{i}.mp4") for i in range(N)] if tmp is not None else []
    w_tiles = writer(out) if args.mode == "frames" else None  # frames: env 0 full size, or 2x2 tiles of envs 0-3
    w_grid = writer(out / f"{args.tag}closeup_grid_3x3.mp4") if args.mode == "closeup" and N >= 9 else None
    gif, phase_log = [], np.zeros((steps, N), np.int64)
    state = {"aimed": False}

    def pol(obs):
        if not state["aimed"]:
            (aim_wide if args.mode == "wide" else aim_task)(u, cam)
            state["aimed"] = True
        return policy(obs)

    def on_step(t, uu):
        pm = uu._phase_machine
        ph = pm.phase.cpu().numpy()
        phase_log[t] = ph
        cube, goal = cube_and_target(uu)
        hit = (torch.linalg.norm(cube - goal, dim=-1) < th.success_r).cpu().numpy()
        fr = grab(uu, "video_cam")
        uvd = project(uu, "video_cam", target_points(uu))
        frames = [overlay(fr[i], uvd[i], [f"phase {ph[i] + 1}/6: {PHASES[ph[i]].upper()}" + ("  (grasp)" if bool(pm.grasp[i]) else ""),
                                          f"{label}   t = {(t + 1) * uu.step_dt:.2f} s"], bool(hit[i])) for i in range(N)]
        for i, w in enumerate(ws):
            w.append_data(frames[i])
        if w_tiles is not None:
            w_tiles.append_data(frames[0] if N == 1 else tile(np.stack([f[::2, ::2] for f in frames[:4]]), 2))
        if w_grid is not None:
            g = tile(np.stack([f[::2, ::2] for f in frames[:9]]), 3)
            w_grid.append_data(g)
            if t % 6 == 0:
                gif.append(g[::4, ::4])

    if args.scripted:
        ctrl.started = False
    res = rollout(run_env, pol, th, on_step=on_step)
    for w in ws:
        w.close()
    if w_tiles is not None:
        w_tiles.close()
    per_env = [{"env": i, **{k: (bool(res[k][i]) if res[k].dtype == torch.bool else round(float(res[k][i]), 4))
                             for k in ("success", "success_5cm", "clean", "lift", "slam", "jam", "table_contact", "dist",
                                       "drops", "final_phase")},
                "reached": [PHASES[k] for k in range(6) if bool(res[f"reached_{PHASES[k]}"][i])],
                "phase_entry_steps": {PHASES[k]: int(np.argmax(phase_log[:, i] == k)) for k in range(6)
                                      if (phase_log[:, i] == k).any()}} for i in range(N)]
    meta = {"task": TASK, "label": label, "checkpoint": args.checkpoint, "scripted": args.scripted, "seed": args.seed,
            "per_env": per_env}
    if args.mode == "frames":
        out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        env.close()
        sys.exit(0)
    import shutil

    outcome = ["success" if e["success"] else ("near_miss" if e["dist"] < 0.10 else "failure") for e in per_env]
    if args.mode == "wide":
        shutil.copyfile(tmp / "env0.mp4", out / f"{args.tag}single_{outcome[0]}_wide.mp4")
    else:
        if w_grid is not None:
            w_grid.close()
            imageio.mimsave(out / f"{args.tag}closeup_grid_3x3.gif", gif, duration=120, loop=0)
        for i in range(N):
            shutil.copyfile(tmp / f"env{i}.mp4", out / f"{args.tag}closeup_env{i}.mp4")
        singles = {}
        for cls in ("success", "near_miss", "failure"):
            idx = [i for i in range(N) if outcome[i] == cls]
            singles[cls] = idx[0] if idx else None
            if idx:
                shutil.copyfile(tmp / f"env{idx[0]}.mp4", out / f"{args.tag}single_{cls}_closeup.mp4")
        meta["singles"] = singles
        audit = ROOT / "results" / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        meta["audit"] = {}
        for i in range(min(args.audit_envs, N)):
            rd = imageio.get_reader(str(tmp / f"env{i}.mp4"))
            frs = [f[::2, ::2] for f in rd]
            rd.close()
            sheets = []
            for k, name in enumerate(PHASES):
                st = np.flatnonzero(phase_log[:, i] == k)
                if len(st) == 0:
                    continue
                pick = sorted({int(st[0]), int(st[len(st) // 2]), int(st[-1])} | (
                    {int(st[len(st) // 4]), int(st[3 * len(st) // 4])} if len(st) >= 5 else set()))
                if len(pick) < 3:  # a 1-2 step phase: add its neighbours so each sheet has >= 3 frames
                    pick = sorted(set(pick) | {max(0, pick[0] - 1), min(steps - 1, pick[-1] + 1)})
                fname = f"{args.tag}env{i}_phase{k + 1}_{name}_steps_{'-'.join(map(str, pick))}.png"
                imageio.imwrite(audit / fname, np.concatenate([frs[s] for s in pick], axis=1))
                sheets.append(fname)
            meta["audit"][i] = sheets
    shutil.rmtree(tmp, ignore_errors=True)
    (out / f"{args.tag}videos.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
