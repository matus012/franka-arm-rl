"""M0 diagnostic: hand-coded approach -> close -> lift on N envs with random cube positions.

Separates "Newton can't grasp this cube" from "PPO found an exploit". Uses the stock IK-abs lift
variant (same scene, cube, gripper and physics as M0; only the arm action is a task-space pose).
Writes results/m0_scripted_grasp.json.

Usage: uv run python scripts/scripted_grasp.py --num_envs 64
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
from lift_rl.instrument import add_eval_sensors
from lift_rl.physics import assert_run_preconditions

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--grasp_depth", type=float, default=0.0, help="EE target z offset from cube center at grasp")
p.add_argument("--out", default=str(ROOT / "logs" / "debug" / "scripted_grasp.json"))
p.add_argument("--video", default="", help="write a 2x2 grid mp4 of envs 0-3 here (debug only)")
p.add_argument("--dump_contacts", action="store_true", help="print env-0 cube contact normals while closing")
p.add_argument("--contact_ke", type=float, default=None, help="physics variant: shape contact stiffness")
p.add_argument("--contact_kd", type=float, default=None, help="physics variant: shape contact damping")
p.add_argument("--finger_armature", type=float, default=None, help="physics variant: finger joint armature")
p.add_argument("--substeps", type=int, default=2, help="physics variant: Newton substeps per 0.01 s sim step")
p.add_argument("--solimp_dmax", type=float, default=None, help="physics variant: MuJoCo solimp dmax (d0 = dmax - 0.04)")
args = p.parse_args()

TASK = "LiftRL-Stock-IK-Newton-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.sim.device = "cuda:0"
env_cfg.episode_length_s = 10.0  # no timeout inside the 250 scripted steps
if args.contact_ke is not None or args.contact_kd is not None or args.substeps != 2:
    from lift_rl.physics import newton_mjwarp_cfg

    env_cfg.sim.physics = newton_mjwarp_cfg(num_substeps=args.substeps, contact_ke=args.contact_ke, contact_kd=args.contact_kd)
if args.finger_armature is not None:
    env_cfg.scene.robot.actuators["panda_hand"].armature = args.finger_armature
if args.solimp_dmax is not None:
    from isaaclab.managers import EventTermCfg

    from lift_rl.physics import set_contact_solimp

    env_cfg.events.solimp = EventTermCfg(func=set_contact_solimp, mode="startup",
                                         params={"solimp": (args.solimp_dmax - 0.04, args.solimp_dmax, 0.001, 0.5, 2.0)})
add_eval_sensors(env_cfg)
if args.video:
    from lift_rl.camera import add_video_camera, aim_video_camera, grab_rgb, tile

    add_video_camera(env_cfg, 320, 240)
assert_run_preconditions(env_cfg)
frames = []

with launch_simulation(env_cfg, {"headless": True, "enable_cameras": bool(args.video)}):
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    if args.video:  # close-up from the front, at each env's cube
        cpos = u.scene["object"].data.root_pos_w.torch
        u.scene["video_cam"].set_world_poses_from_view(
            cpos + torch.tensor([0.40, 0.0, 0.10], device=u.device), cpos + torch.tensor([0.0, 0.0, 0.04], device=u.device)
        )
    robot, obj = u.scene["robot"], u.scene["object"]
    arm = u.action_manager.get_term("arm_action")
    N = u.num_envs

    def cube_b() -> torch.Tensor:
        return obj.data.root_pos_w.torch - robot.data.root_pos_w.torch  # robot base has identity rotation

    # settle: hold the current pose, gripper open
    ee_pos, ee_quat = arm._compute_frame_pose()
    hold = torch.cat([ee_pos, ee_quat, torch.ones(N, 1, device=u.device)], dim=1)
    for _ in range(10):
        env.step(hold)
    rest_z = obj.data.root_pos_w.torch[:, 2].clone()
    c0 = cube_b().clone()
    # top-down grasp: hand z-axis pointing at -z world (180 deg about x, quaternion xyzw). The stock
    # default pose has the hand tilted ~45 deg, which rams the table when descending.
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=u.device).repeat(N, 1)

    print("[grasp] hand-table sensors:", u.scene["contact_hand_table"].sensor_names, u.scene["contact_hand_table"].filter_object_names)
    print("[grasp] cube-hand filters:", u.scene["contact_cube_hand"].filter_object_names)
    from isaaclab_newton.physics import NewtonManager as _M

    print("[grasp] mjw geom_solimp[0,:3]:", _M._solver.mjw_model.geom_solimp.numpy()[0, :3].tolist())
    print("[grasp] rest_z:", rest_z[:4].tolist(), "ee quat:", quat[0].tolist())
    z_hist, palm_max, fing_both, table_max = [], torch.zeros(N, device=u.device), torch.zeros(N, device=u.device), torch.zeros(N, device=u.device)
    t0 = time.time()
    integ = torch.zeros(N, 3, device=u.device)
    for t in range(250):
        freeze = False
        if t < 60:  # above the cube
            tgt, g = c0 + torch.tensor([0, 0, 0.12], device=u.device), 1.0
        elif t < 135:  # descend on a ramp (a step change overshoots into the cube), then settle
            s = min((t - 60) / 40.0, 1.0)
            tgt, g = c0 + torch.tensor([0, 0, 0.12 * (1 - s) + args.grasp_depth * s], device=u.device), 1.0
        elif t < 165:  # close; integral frozen so the hand does not creep while the fingers close
            tgt, g, freeze = c0 + torch.tensor([0, 0, args.grasp_depth], device=u.device), -1.0, True
        else:  # lift on a ramp
            s = min((t - 165) / 50.0, 1.0)
            tgt, g = c0 + torch.tensor([0, 0, args.grasp_depth * (1 - s) + 0.20 * s], device=u.device), -1.0
        # integral correction: the IK action tracks with a steady offset (gravity vs PD, Newton
        # jacobian), so command tgt + I where I accumulates the tracking error
        ep, _ = arm._compute_frame_pose()
        if not freeze:
            integ = (integ + 0.1 * (tgt - ep)).clamp(-0.15, 0.15)
        a = torch.cat([tgt + integ, quat, torch.full((N, 1), g, device=u.device)], dim=1)
        env.step(a)
        if args.video and t % 2 == 0:
            frames.append(tile(grab_rgb(env)[:4], 2))
        if t in (59, 100, 134, 164, 200, 249):
            ep, _ = arm._compute_frame_pose()
            hand = robot.data.body_pos_w.torch[0, robot.body_names.index("panda_hand")] - robot.data.root_pos_w.torch[0]
            fm0 = u.scene["contact_cube_hand"].data.force_matrix_w.torch.reshape(N, 3, 3).norm(dim=-1)[0]
            print(f"[grasp] t={t} tgt={tgt[0].tolist()} ee={ep[0].tolist()} hand={hand.tolist()} "
                  f"cube={cube_b()[0].tolist()} fingers={robot.data.joint_pos.torch[0, -2:].tolist()} "
                  f"F(hand,lf,rf)={fm0.tolist()}")
        if args.dump_contacts and t in (136, 138, 140, 143, 146, 150, 155):
            from isaaclab_newton.physics import NewtonManager as M

            c = M._contacts
            n = int(c.rigid_contact_count.numpy()[0])
            s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
            nrm = c.rigid_contact_normal.numpy()[:n]
            lab = M._model.shape_label
            cube_i = [i for i, l in enumerate(lab) if l.endswith("env_0/Object/collisions/collisions")][0]
            rows = [(lab[b].split("/")[-3] if a == cube_i else lab[a].split("/")[-3], nrm[k].round(3).tolist())
                    for k, (a, b) in enumerate(zip(s0, s1)) if cube_i in (a, b)]
            print(f"[contacts] t={t} fingers={robot.data.joint_pos.torch[0, -2:].tolist()} n={len(rows)} {rows}")
        z_hist.append((obj.data.root_pos_w.torch[:, 2] - rest_z).clone())
        fm = u.scene["contact_cube_hand"].data.force_matrix_w.torch.reshape(N, 3, 3).norm(dim=-1)  # hand, lf, rf
        palm_max = torch.maximum(palm_max, fm[:, 0])
        if t >= 165:
            fing_both += ((fm[:, 1] > 0.5) & (fm[:, 2] > 0.5)).float()
        ht = u.scene["contact_hand_table"].data.force_matrix_w.torch.reshape(N, -1, 3).norm(dim=-1).amax(dim=1)
        table_max = torch.maximum(table_max, ht)

    z = torch.stack(z_hist, dim=1)
    lifted = z[:, -1] >= 0.05
    res = {
        "num_envs": N,
        "seed": args.seed,
        "grasp_depth": args.grasp_depth,
        "contact_ke": args.contact_ke,
        "contact_kd": args.contact_kd,
        "finger_armature": args.finger_armature,
        "solimp_dmax": args.solimp_dmax,
        "substeps": args.substeps,
        "lifted_at_end": int(lifted.sum()),
        "lift_rate": float(lifted.float().mean()),
        "final_dz_median_m": float(z[:, -1].median()),
        "final_dz_quantiles_m": [float(q) for q in torch.quantile(z[:, -1], torch.tensor([0.1, 0.5, 0.9], device=u.device))],
        "rest_z_m": float(rest_z.mean()),
        "palm_cube_force_max_N_median": float(palm_max.median()),
        "both_fingers_contact_frac_during_lift_mean": float((fing_both / 85).mean()),
        "hand_table_force_max_N_quantiles": [float(q) for q in torch.quantile(table_max, torch.tensor([0.5, 0.9, 1.0], device=u.device))],
        "nan_guard_events": u.nan_guard_events,
        "wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(res, indent=2))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    if args.video:
        import imageio

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=25)
        imageio.imwrite(str(Path(args.video).with_suffix(".png")), frames[len(frames) * 13 // 25])
    env.close()
