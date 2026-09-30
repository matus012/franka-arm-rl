"""Section 14: fully scripted reference demo (no RL). A phase state machine drives the fingertip center through
hover -> descend -> close -> lift -> carry -> hold on LiftRL-Staged-v0 (same scene, fix-2 physics, cube
position + yaw randomization, target sampling and 7 joint-position + binary-gripper action space as the policies).

Controller, per 50 Hz control step:
  - each phase gives a task-space target for the fingertip center (TCP = panda_hand + 10.34 cm on its z-axis) and a
    hand orientation (pointing down, closing axis on the nearest pair of cube faces, yaw mod 90 deg). The position
    setpoint moves toward the phase goal on a straight line at the phase's speed cap with <= 2 m/s^2 acceleration; the
    orientation setpoint turns toward its goal at <= 1.5 rad/s.
  - damped-least-squares IK on the robot Jacobian (TCP-shifted):  dq = J^T (J J^T + lambda^2 I)^-1 e, plus a small
    null-space pull toward the default arm pose.
  - joint-position action = q_measured + dq + feed-forward terms divided by the joint stiffness: gravity (from the link
    COM Jacobians and masses; Newton has no gravity-compensation primitive), the held cube's weight, M(q) qdd_des for
    the setpoint acceleration, the PD damping for the setpoint velocity; plus extra damping on joints 1, 2 and 4
    (the stock arm PD, Kp 80 / Kd 4 with gravity on, is slow and lightly damped). Mapped to the env's action scale
    (0.5, default-pose offset). The gripper command is binary (open / close), as for the policies.
  The IK-abs action variant is NOT used (it had a 5-8 cm steady offset in run 1). The controller reads the sim state
  (cube pose, target, contacts, joint state), not the policy observation.

Modes:
  --mode eval         1,000 episodes (seed 12345) through lift_rl.evaluation.rollout + the unchanged detectors,
                      plus a per-phase failure breakdown -> results/scripted_demo_eval.json
  --mode video        N episodes (default 9, seed 2024) with a close-up camera framing cube + target and a wide
                      camera; phase name + target marker overlaid; real time (50 fps). Writes videos/scripted_demo/
                      (closeup_env<i>.mp4, closeup_grid_3x3.mp4/.gif, single_success_closeup.mp4,
                      single_success_wide.mp4) and >= 3 audit frames per phase to results/audit/
  --mode frames       worker for the side-by-side: --policy scripted|rl, 4 envs as 2x2 close-up tiles -> mp4
  --mode side_by_side scripted (left) vs probe-5 model_250 (right), same seed -> videos/scripted_demo/side_by_side.mp4
Usage: uv run python scripts/scripted_demo.py --mode eval
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASK = "LiftRL-Staged-v0"
RL_TASK = "LiftRL-Staged-AutoClose-v0"  # probe 5's training env (same scene/randomization, scripted close trigger)
RL_CKPT = "logs/rsl_rl/lift_rl/2026-09-29_10-26-24_r2_probe5/model_250.pt"
OUT = ROOT / "videos" / "scripted_demo"

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("eval", "video", "frames", "side_by_side"), required=True)
p.add_argument("--num_envs", type=int, default=None)
p.add_argument("--num_episodes", type=int, default=1000)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--policy", choices=("scripted", "rl"), default="scripted")
p.add_argument("--checkpoint", default=RL_CKPT)
p.add_argument("--out", default=None)
p.add_argument("--tag", default="")
p.add_argument("--init", default=None, help="frames mode, rl side: replay the scripted side's initial cube poses and targets")
p.add_argument("--debug_env", type=int, default=-1)
args = p.parse_args()

if args.mode == "side_by_side":
    import imageio
    import numpy as np

    seed = args.seed if args.seed is not None else 2024
    OUT.mkdir(parents=True, exist_ok=True)
    parts, metas = [], []
    for side, pol in (("left", "scripted"), ("right", "rl")):
        f = OUT / f"_sbs_{side}.mp4"
        # the RL env (scripted close trigger) draws its random numbers differently, so the same seed alone gives other
        # cube poses/targets: the right side replays the left side's initial cube pose and target instead
        extra = ["--init", str(OUT / "_sbs_left.init.json")] if pol == "rl" else []
        subprocess.run([sys.executable, "-u", __file__, "--mode", "frames", "--policy", pol, "--seed", str(seed),
                        "--out", str(f)] + extra, check=True)
        parts.append(f)
        metas.append(json.loads(f.with_suffix(".json").read_text()))
    ra, rb = (imageio.get_reader(str(f)) for f in parts)
    w = imageio.get_writer(str(OUT / "side_by_side.mp4"), fps=50, codec="libx264", quality=8, macro_block_size=8)
    for fa, fb in zip(ra, rb):
        bar = np.full((fa.shape[0], 8, 3), 255, np.uint8)
        w.append_data(np.concatenate([fa, bar, fb], axis=1))
    w.close()
    ra.close()
    rb.close()
    meta = {"seed": seed, "left": metas[0], "right": metas[1], "layout": "envs 0-3 as 2x2 close-up tiles per side"}
    (OUT / "side_by_side.json").write_text(json.dumps(meta, indent=2))
    for f in parts:
        f.unlink()
        f.with_suffix(".json").unlink()
    (OUT / "_sbs_left.init.json").unlink()
    print(json.dumps(meta, indent=1))
    sys.exit(0)

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import lift_rl  # noqa: E402,F401
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error, euler_xyz_from_quat, matrix_from_quat, quat_from_angle_axis, quat_from_euler_xyz, quat_mul)
from isaaclab_newton.renderers import NewtonWarpRendererCfg  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.camera import tile  # noqa: E402
from lift_rl.evaluation import EPISODE_STEPS, contact_readings, cube_and_target, make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import CUBE_REST_Z, Thresholds, summarize  # noqa: E402
from lift_rl.physics import assert_on_cuda, assert_run_preconditions  # noqa: E402
from lift_rl.staged import yaw_mod90_error  # noqa: E402

PHASES = ("hover", "descend", "close", "lift", "carry", "hold")
HOVER, DESCEND, CLOSE, LIFT, CARRY, HOLD = range(6)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


class ScriptedController:
    """Phase state machine + DLS IK. Called like a policy (obs ignored; reads the sim state)."""

    HOVER_DZ = 0.10  # fingertip center above the cube top
    GRASP_MIN_Z = 0.01  # fingertip center >= 1 cm above the table
    LIFT_DZ = 0.10
    V_APPROACH, V_DESCEND, V_LIFT, V_CARRY = 1.00, 0.10, 0.15, 0.25  # setpoint speed caps [m/s] (approach: not capped by 14)
    XY_TOL, YAW_TOL = 0.005, math.radians(5.0)
    CLOSE_MAX_STEPS = 25  # 0.5 s
    HOVER_STEPS = 5  # the hover gate must hold for 5 consecutive steps (0.1 s), not only pass through once
    PAD_STEPS = 3  # both pads touching for 3 consecutive steps (0.06 s)
    SETTLE_STEPS = 15  # descent: wait at most 0.3 s for the TCP to reach grasp height after the setpoint did
    LAMBDA, NULL_GAIN = 0.02, 0.02
    W_MAX = 2.5  # orientation setpoint rate limit [rad/s] (the start pose is tilted ~45 deg)
    A_MAX = 2.0  # setpoint acceleration limit [m/s^2]: smooth starts/stops (+ acceleration feed-forward below)
    # extra damping through the position target on the base joint 1 and the pitch joints 2 and 4, which carry the whole
    # arm (joint 2: ~3 kg m^2 on Kp 80 / Kd 4 -> ~1.2 s period, damping ratio ~0.13): Kd_eff = Kd + DAMP * Kp. Not on
    # the light joints 3, 5-7 and at most 0.1 on joint 1: beyond that the one-step-late velocity feedback goes unstable
    DAMP = (0.1, 0.2, 0.0, 0.2, 0.0, 0.0, 0.0)
    TCP_OFFSET = 0.1034

    def __init__(self, env, th: Thresholds):
        self.u = u = env.unwrapped
        self.th = th
        self.robot = r = u.scene["robot"]
        self.obj = u.scene["object"]
        self.N, self.dev = u.num_envs, u.device
        self.hand = r.body_names.index("panda_hand")
        self.jac_body = self.hand - 1 if r.is_fixed_base else self.hand
        self.arm_ids = r.find_joints("panda_joint.*")[0]
        nb = getattr(r, "num_base_dofs", 0)
        self.jac_cols = [j + nb for j in self.arm_ids]
        self.q_default = _t(r.data.default_joint_pos)[:, self.arm_ids].clone()
        lim = _t(r.data.soft_joint_pos_limits)[:, self.arm_ids]
        self.q_lo, self.q_hi = lim[..., 0], lim[..., 1]
        self.kp = _t(r.data.joint_stiffness)[:, self.arm_ids].clamp_min(1.0)
        self.kd = _t(r.data.joint_damping)[:, self.arm_ids]
        self.damp = torch.tensor(self.DAMP, device=self.dev)
        term = u.action_manager.get_term("arm_action")
        self.a_scale, self.a_offset = term._scale, term._offset
        m = _t(r.data.body_mass)
        self.link_mass = m[:, 1:] if r.is_fixed_base else m  # Newton drops the fixed root's Jacobian row
        self.cube_mass = _t(self.obj.data.body_mass).reshape(self.N, -1).sum(1)
        self.started = False

    # ---------------------------------------------------------------- state
    def _reset(self):
        N, dev = self.N, self.dev
        self.phase = torch.zeros(N, dtype=torch.long, device=dev)
        self.phase_t = torch.zeros(N, dtype=torch.long, device=dev)
        self.t = 0
        self.entered = torch.full((N, 6), -1, dtype=torch.long, device=dev)  # step each phase was entered
        self.entered[:, HOVER] = 0
        self.sp = self._tcp().clone()
        self.v = torch.zeros(N, 3, device=dev)  # setpoint velocity
        self.q_sp = _t(self.robot.data.body_link_quat_w)[:, self.hand].clone()  # orientation setpoint
        self.pad_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.hover_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.close_touched = torch.zeros(N, dtype=torch.bool, device=dev)
        self.grasp_z = torch.zeros(N, device=dev)
        self.lift_goal = torch.zeros(N, 3, device=dev)
        self.offset = torch.zeros(N, 3, device=dev)  # cube center - TCP, measured at the end of the lift
        self.at_goal_t = torch.zeros(N, dtype=torch.long, device=dev)
        self.palm_phase = torch.full((N,), -1, dtype=torch.long, device=dev)  # phase at the first palm-cube contact
        self.started = True

    def _tcp(self):
        # from the hand link pose (the ee_frame sensor is stale on the reset step); same point as ee_frame's target
        r = self.robot
        R = matrix_from_quat(_t(r.data.body_link_quat_w)[:, self.hand])
        return _t(r.data.body_link_pos_w)[:, self.hand] + R[:, :, 2] * self.TCP_OFFSET - self.u.scene.env_origins

    def _advance(self, mask, to):
        m = mask & (self.phase == to - 1)
        self.phase = torch.where(m, torch.full_like(self.phase, to), self.phase)
        self.phase_t = torch.where(m, torch.zeros_like(self.phase_t), self.phase_t)
        self.entered[:, to] = torch.where(m & (self.entered[:, to] < 0), torch.full_like(self.phase, self.t), self.entered[:, to])
        return m

    # ---------------------------------------------------------------- control
    @torch.inference_mode()
    def __call__(self, obs=None) -> torch.Tensor:
        if not self.started:
            self._reset()
        u, r, th = self.u, self.robot, self.th
        p = self._tcp()
        qh = _t(r.data.body_link_quat_w)[:, self.hand]
        R = matrix_from_quat(qh)
        cube, goal = cube_and_target(u)
        _, _, cube_yaw = euler_xyz_from_quat(_t(self.obj.data.root_quat_w))
        c = contact_readings(u)
        both = (c["lf"] > th.contact_n) & (c["rf"] > th.contact_n)
        self.palm_phase = torch.where((self.palm_phase < 0) & (c["palm"] > th.contact_n), self.phase, self.palm_phase)

        # hand yaw: closing axis (hand y) on the nearest pair of cube faces, the rotation <= 45 deg from now
        psi_now = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        k = torch.round((psi_now - cube_yaw) / (math.pi / 2))
        psi = cube_yaw + k * (math.pi / 2)
        z0 = torch.zeros_like(psi)
        q_des = quat_from_euler_xyz(torch.full_like(psi, math.pi), z0, psi)  # hand z-axis straight down
        yaw_err = yaw_mod90_error(R[:, :, 1], cube_yaw)
        tilt = torch.arccos((-R[:, 2, 2]).clamp(-1.0, 1.0))

        # ---- phase transitions (evaluated on the current state)
        ph = self.phase
        cube_top = cube[:, 2] + CUBE_REST_Z
        hover_goal = torch.stack([cube[:, 0], cube[:, 1], cube_top + self.HOVER_DZ], 1)
        xy_err = torch.linalg.norm(p[:, :2] - cube[:, :2], dim=-1)
        # section 14 gate: hover setpoint reached, xy error < 5 mm, yaw error < 5 deg
        ok = (torch.linalg.norm(self.sp - hover_goal, dim=-1) < 2e-3) & (xy_err < self.XY_TOL) & (yaw_err < self.YAW_TOL)
        self.hover_run = torch.where((ph == HOVER) & ok, self.hover_run + 1, torch.zeros_like(self.hover_run))
        m = self._advance((ph == HOVER) & (self.hover_run >= self.HOVER_STEPS), DESCEND)
        self.grasp_z = torch.where(m, cube[:, 2].clamp_min(self.GRASP_MIN_Z), self.grasp_z)

        sp_down = (ph == DESCEND) & ((self.sp[:, 2] - self.grasp_z).abs() < 1e-4)
        self.at_goal_t = torch.where(sp_down, self.at_goal_t + 1, torch.zeros_like(self.at_goal_t))
        reached = (p[:, 2] - self.grasp_z).abs() < 0.004
        self._advance(sp_down & (reached | (self.at_goal_t > self.SETTLE_STEPS)), CLOSE)

        self.pad_run = torch.where((self.phase == CLOSE) & both, self.pad_run + 1, torch.zeros_like(self.pad_run))
        touched = self.pad_run >= self.PAD_STEPS
        self.close_touched |= (self.phase == CLOSE) & touched
        m = self._advance((ph == CLOSE) & (touched | (self.phase_t >= self.CLOSE_MAX_STEPS)), LIFT)
        self.lift_goal = torch.where(m[:, None], torch.cat([p[:, :2], (self.grasp_z + self.LIFT_DZ)[:, None]], 1), self.lift_goal)

        lifted = (ph == LIFT) & (torch.linalg.norm(self.sp - self.lift_goal, dim=-1) < 1e-4) & (torch.linalg.norm(p - self.lift_goal, dim=-1) < 0.01)
        m = self._advance(lifted, CARRY)
        self.offset = torch.where(m[:, None], cube - p, self.offset)

        carry_goal = goal - self.offset
        self._advance((ph == CARRY) & (torch.linalg.norm(self.sp - carry_goal, dim=-1) < 1e-4), HOLD)

        # ---- setpoint for the (possibly new) phase
        ph = self.phase
        grasp_pt = torch.stack([cube[:, 0], cube[:, 1], self.grasp_z], 1)
        goal_pt = torch.where((ph == HOVER)[:, None], hover_goal, grasp_pt)
        goal_pt = torch.where((ph == CLOSE)[:, None], torch.cat([self.sp[:, :2], self.grasp_z[:, None]], 1), goal_pt)
        goal_pt = torch.where((ph == LIFT)[:, None], self.lift_goal, goal_pt)
        goal_pt = torch.where((ph >= CARRY)[:, None], carry_goal, goal_pt)
        vmax = torch.tensor([self.V_APPROACH, self.V_DESCEND, self.V_DESCEND, self.V_LIFT, self.V_CARRY, self.V_CARRY],
                            device=self.dev)[ph]  # [m/s]
        dt = u.step_dt
        dn = ph == DESCEND  # descent is vertical: xy follows the (resting) cube, z moves at <= 0.10 m/s
        self.sp[:, :2] = torch.where(dn[:, None], goal_pt[:, :2], self.sp[:, :2])
        d = goal_pt - self.sp
        d[:, :2] = torch.where(dn[:, None], 0.0, d[:, :2])
        n = torch.linalg.norm(d, dim=-1, keepdim=True)
        # speed profile: cap vmax, accelerate/brake at <= A_MAX (brake so that it stops on the goal)
        speed = torch.minimum(vmax[:, None], torch.sqrt(2 * self.A_MAX * n))
        v_des = d / n.clamp_min(1e-9) * speed
        v_prev = self.v.clone()
        dv = v_des - self.v
        self.v = self.v + dv * torch.clamp(self.A_MAX * dt / torch.linalg.norm(dv, dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
        step = self.v * dt
        snap = torch.linalg.norm(step, dim=-1, keepdim=True) >= n  # arrive exactly
        self.sp = torch.where(snap, goal_pt, self.sp + step)
        self.sp[:, :2] = torch.where(dn[:, None], goal_pt[:, :2], self.sp[:, :2])
        self.v = torch.where(snap, torch.zeros_like(self.v), self.v)
        a_sp = (self.v - v_prev) / dt  # setpoint acceleration (piecewise constant, |a| <= A_MAX)
        v_sp = self.v

        # ---- DLS IK on the TCP Jacobian + gravity feed-forward
        J = _t(r.data.body_link_jacobian_w)[:, self.jac_body][:, :, self.jac_cols].clone()  # (N,6,7) world
        rr = R[:, :, 2] * self.TCP_OFFSET  # hand origin -> TCP, world
        skew = torch.zeros(self.N, 3, 3, device=self.dev)
        skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 0] = -rr[:, 2], rr[:, 1], rr[:, 2]
        skew[:, 1, 2], skew[:, 2, 0], skew[:, 2, 1] = -rr[:, 0], -rr[:, 1], rr[:, 0]
        J[:, :3] = J[:, :3] - torch.bmm(skew, J[:, 3:])
        e_pos = self.sp - p
        # orientation setpoint: rotates toward q_des at <= W_MAX
        _, r_err = compute_pose_error(p, self.q_sp, p, q_des, rot_error_type="axis_angle")
        ang = torch.linalg.norm(r_err, dim=-1)
        w_step = r_err * torch.clamp(self.W_MAX * dt / ang.clamp_min(1e-9), max=1.0)[:, None]
        self.q_sp = quat_mul(quat_from_angle_axis(torch.linalg.norm(w_step, dim=-1), w_step / torch.linalg.norm(w_step, dim=-1, keepdim=True).clamp_min(1e-9)), self.q_sp)
        w_sp = w_step / dt
        _, e_rot = compute_pose_error(p, qh, p, self.q_sp, rot_error_type="axis_angle")
        e = torch.cat([e_pos.clamp(-0.05, 0.05), e_rot], 1)
        JJt = torch.bmm(J, J.transpose(1, 2)) + (self.LAMBDA**2) * torch.eye(6, device=self.dev)
        Jpinv = torch.bmm(J.transpose(1, 2), torch.linalg.inv(JJt))  # (N,7,6)
        q = _t(r.data.joint_pos)[:, self.arm_ids]
        dq = torch.bmm(Jpinv, e[:, :, None])[..., 0]
        null = torch.eye(7, device=self.dev) - torch.bmm(Jpinv, J)
        dq = dq + self.NULL_GAIN * torch.bmm(null, (self.q_default - q)[:, :, None])[..., 0]
        # Newton has no gravity-compensation primitive: tau_g = sum_b J_com,b^T (m_b g), over every link's COM Jacobian
        Jc = _t(r.data.body_com_jacobian_w)[:, :, 2][:, :, self.jac_cols]  # (N, B', 7): d(COM z)/dq
        tau_g = 9.81 * torch.einsum("nb,nbj->nj", self.link_mass, Jc)
        # + the cube's weight at the fingertips once it is held (both pads, lift or later)
        held = (ph >= LIFT) & both
        tau_g = tau_g + torch.where(held[:, None], 9.81 * self.cube_mass[:, None] * J[:, 2], 0.0)
        # velocity feed-forward: the PD's damping term Kd*qdot would otherwise make the arm lag a moving setpoint
        qd_des = torch.bmm(Jpinv, torch.cat([v_sp, w_sp], 1)[:, :, None])[..., 0]
        qd = _t(r.data.joint_vel)[:, self.arm_ids]
        # acceleration feed-forward M(q) qdd_des / Kp (the heavy links otherwise lag and overshoot every start/stop)
        qdd_des = torch.bmm(Jpinv[:, :, :3], a_sp[:, :, None])
        M = _t(r.data.mass_matrix)[:, self.jac_cols][:, :, self.jac_cols]
        tau_a = torch.bmm(M, qdd_des)[..., 0]
        q_cmd = (q + dq).clamp(self.q_lo, self.q_hi) + (tau_g + tau_a) / self.kp + (self.kd / self.kp + self.damp) * qd_des - self.damp * qd
        a_arm = (q_cmd - self.a_offset) / self.a_scale
        grip = torch.where(ph >= CLOSE, -1.0, 1.0)[:, None]

        if args.debug_env >= 0 and self.t % 5 == 0:
            i = args.debug_env
            print(f"[dbg] t={self.t:3d} ph={PHASES[int(ph[i])]:7s} p=({p[i,0]:.3f},{p[i,1]:.3f},{p[i,2]:.3f}) "
                  f"sp=({self.sp[i,0]:.3f},{self.sp[i,1]:.3f},{self.sp[i,2]:.3f}) cube=({cube[i,0]:.3f},{cube[i,1]:.3f},{cube[i,2]:.3f}) "
                  f"goal=({goal[i,0]:.3f},{goal[i,1]:.3f},{goal[i,2]:.3f}) xy={xy_err[i]*1000:.1f}mm yaw={math.degrees(yaw_err[i]):.1f} "
                  f"tilt={math.degrees(tilt[i]):.1f} lf={c['lf'][i]:.1f} rf={c['rf'][i]:.1f} palm={c['palm'][i]:.1f} "
                  f"tbl={c['robot_table'][i]:.1f}", flush=True)
        self.phase_t += 1
        self.t += 1
        return torch.cat([a_arm, grip], 1)


# -------------------------------------------------------------------------------- video helpers
CUBE_EDGE = 2 * CUBE_REST_Z
DIR = torch.tensor([0.34, 0.40, 0.24])
DIR = DIR / DIR.norm()
try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 34)
    FONT_S = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
except OSError:
    FONT = ImageFont.load_default(size=34)
    FONT_S = ImageFont.load_default(size=26)


def cam_cfg(focal: float, w: int = 1280, h: int = 720) -> CameraCfg:
    return CameraCfg(prim_path="{ENV_REGEX_NS}/VideoCam", update_latest_camera_pose=True, data_types=["rgb"],
                     spawn=sim_utils.PinholeCameraCfg(focal_length=focal, focus_distance=400.0, horizontal_aperture=20.955,
                                                      clipping_range=(0.02, 20.0)),
                     width=w, height=h, renderer_cfg=NewtonWarpRendererCfg())


def aim_task(u, cam) -> None:
    """Close-up framing of the cube's start position AND the target, table surface in view."""
    cube, goal = cube_and_target(u)
    c0 = cube.clone()
    c0[:, 2] = CUBE_REST_Z
    look = 0.5 * (c0 + goal) + torch.tensor([0.0, 0.0, 0.07], device=u.device)  # the hand sits above the cube
    dist = (0.6 + 1.6 * torch.linalg.norm(goal - c0, dim=-1)).clamp(0.8, 1.7)
    o = u.scene.env_origins
    cam.set_world_poses_from_view(o + look + dist[:, None] * DIR.to(u.device), o + look)


WIDE_EYE, WIDE_LOOK = (1.35, 1.00, 0.85), (0.35, 0.0, 0.25)  # farther and higher than lift_rl.camera's EYE/TARGET


def aim_wide(u, cam) -> None:
    o = u.scene.env_origins
    cam.set_world_poses_from_view(o + torch.tensor(WIDE_EYE, device=u.device), o + torch.tensor(WIDE_LOOK, device=u.device))


def grab(u, name: str) -> np.ndarray:
    rgb = _t(u.scene[name].data.output["rgb"])
    return rgb[..., :3].to(torch.uint8).cpu().numpy()


def project(u, name: str, pts_env: torch.Tensor) -> np.ndarray:
    """(N,K,3) env-frame points -> (N,K,3) pixel u, v, depth for camera `name`."""
    cam = u.scene[name]
    pos = _t(cam.data.pos_w)
    Rw = matrix_from_quat(_t(cam.data.quat_w_ros))  # ROS: +z forward, +x right, +y down
    K = _t(cam.data.intrinsic_matrices)
    pw = pts_env + u.scene.env_origins[:, None]
    pc = torch.einsum("nji,nkj->nki", Rw, pw - pos[:, None])
    uvw = torch.einsum("nij,nkj->nki", K, pc)
    return torch.cat([uvw[..., :2] / uvw[..., 2:3].clamp_min(1e-6), pc[..., 2:3]], -1).cpu().numpy()


_CORNERS = torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32) * (CUBE_EDGE / 2)
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if int((_CORNERS[a] != _CORNERS[b]).sum()) == 1]


def target_points(u) -> torch.Tensor:
    _, goal = cube_and_target(u)
    pts = goal[:, None] + _CORNERS.to(u.device)[None]
    floor = goal.clone()
    floor[:, 2] = 0.0
    return torch.cat([pts, goal[:, None], floor[:, None]], 1)  # 8 corners, center, center projected to the table


def overlay(frame: np.ndarray, uvd: np.ndarray, lines: list[str], hit: bool) -> np.ndarray:
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im, "RGBA")
    col = (40, 230, 90, 255) if hit else (255, 60, 200, 255)
    if (uvd[:, 2] > 0).all():
        for k in range(0, 1000, 2):  # dashed drop line from the target to the table
            a, b = uvd[8, :2], uvd[9, :2]
            s0, s1 = k / 40, (k + 1) / 40
            if s0 >= 1:
                break
            dr.line([tuple(a + (b - a) * s0), tuple(a + (b - a) * min(s1, 1))], fill=col[:3] + (150,), width=2)
        for a, b in _EDGES:
            dr.line([tuple(uvd[a, :2]), tuple(uvd[b, :2])], fill=col, width=4)
        cx, cy = uvd[8, :2]
        dr.text((cx + 26, cy - 40), "target", font=FONT_S, fill=col)
    y = 14
    for j, s in enumerate(lines):
        f = FONT if j == 0 else FONT_S
        bb = dr.textbbox((18, y), s, font=f)
        dr.rectangle([bb[0] - 8, bb[1] - 6, bb[2] + 8, bb[3] + 6], fill=(0, 0, 0, 150))
        dr.text((18, y), s, font=f, fill=(255, 255, 255, 255))
        y = bb[3] + 14
    return np.asarray(im)


def writer(path: Path, fps: int = 50):
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=8)


# -------------------------------------------------------------------------------- run
seed = args.seed if args.seed is not None else (12345 if args.mode == "eval" else 2024)
N = args.num_envs or {"eval": 1000, "video": 9, "frames": 4}[args.mode]
task = RL_TASK if args.policy == "rl" else TASK
cfg = make_eval_cfg(load_cfg_from_registry(task, "env_cfg_entry_point"), N, seed)
video = args.mode in ("video", "frames")
if video:
    cfg.scene.video_cam = cam_cfg(18.0)
    if args.mode == "video":
        wide = cam_cfg(15.0)
        wide.prim_path = "{ENV_REGEX_NS}/WideCam"
        cfg.scene.wide_cam = wide
assert_run_preconditions(cfg)
th = Thresholds()

with launch_simulation(cfg, {"headless": True, "enable_cameras": video}):
    env = gym.make(task, cfg=cfg)
    u = env.unwrapped
    if args.policy == "rl":
        from lift_rl.policy import load_policy

        run_env, policy, module = load_policy(env, task, args.checkpoint)
        assert_on_cuda(env, module)
        ctrl = None
    else:
        run_env = env
        ctrl = ScriptedController(env, th)
        policy = ctrl
        assert str(u.device).startswith("cuda"), u.device

    if args.mode == "eval":
        import time

        t0 = time.time()
        batches, extra = [], []
        while sum(b["success"].numel() for b in batches) < args.num_episodes:
            ctrl.started = False
            batches.append(rollout(run_env, policy, th))
            cube, _ = cube_and_target(u)
            extra.append({"palm_phase": ctrl.palm_phase.clone(), "phase": ctrl.phase.clone(), "entered": ctrl.entered.clone(), "close_touched": ctrl.close_touched.clone(),
                          "cube_z": cube[:, 2].clone(), "both_pads": ((contact_readings(u)["lf"] > th.contact_n) & (contact_readings(u)["rf"] > th.contact_n)).clone()})
        n = args.num_episodes
        out = {k: torch.cat([b[k] for b in batches])[:n] for k in batches[0]}
        ex = {k: torch.cat([b[k] for b in extra])[:n] for k in extra[0]}
        s = summarize(out)
        fail = ~out["success"]
        ph_end = ex["phase"]
        held = ex["both_pads"] & (ex["cube_z"] > CUBE_REST_Z + th.off_table_dz)
        breakdown = {}
        for k, name in enumerate(PHASES):
            m = fail & (ph_end == k)
            row = {"count": int(m.sum()), "pct_of_episodes": round(100 * float(m.float().mean()), 2)}
            if k == HOLD:
                row["cube_not_held_at_end"] = int((m & ~held).sum())
                row["held_but_dist_ge_3cm"] = int((m & held).sum())
                row["held_dist_median_m"] = float(out["dist"][m & held].median()) if bool((m & held).any()) else None
            if k >= LIFT:
                row["close_timed_out_without_both_pads"] = int((m & ~ex["close_touched"]).sum())
            breakdown[f"ended_in_{name}"] = row
        reached = {name: round(100 * float((ex["entered"][:, k] >= 0).float().mean()), 2) for k, name in enumerate(PHASES)}
        entry = {name: (float(ex["entered"][:, k][ex["entered"][:, k] >= 0].float().median()) * u.step_dt
                        if bool((ex["entered"][:, k] >= 0).any()) else None) for k, name in enumerate(PHASES)}
        not_clean = ~out["clean"]
        clean_fail = {k: int((not_clean & ~out[k]).sum()) for k in ("clean_c1_no_table_hit", "clean_c2_top_down",
                                                                  "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm")}
        res = {"task": TASK, "controller": "scripts/scripted_demo.py (phase state machine + DLS IK, joint-position actions)",
               "eval_seed": seed, "num_envs": N, "thresholds": th.to_dict(), "nan_guard_events": u.nan_guard_events,
               "wall_s": round(time.time() - t0, 1), **s,
               "per_phase": {"failures_by_phase_at_episode_end": breakdown, "reached_phase_pct": reached,
                             "median_phase_entry_time_s": entry,
                             "close_both_pads_pct": round(100 * float(ex["close_touched"].float().mean()), 2),
                             "not_clean_by_criterion": clean_fail,
                             "first_palm_contact_in_phase": {name: int((ex["palm_phase"] == k).sum()) for k, name in enumerate(PHASES)}}}
        (ROOT / "results").mkdir(exist_ok=True)
        dst = Path(args.out) if args.out else ROOT / "results" / "scripted_demo_eval.json"
        dst.write_text(json.dumps(res, indent=2))
        print(json.dumps({k: res[k] for k in ("success", "success_5cm", "clean", "lift", "slam", "jam", "final_dist_median_m")}, indent=1))
        print(json.dumps(res["per_phase"], indent=1))
        print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
        env.close()
        sys.exit(0)

    # ------------------------------------------------------------ video / frames
    OUT.mkdir(parents=True, exist_ok=True)
    cam = u.scene["video_cam"]
    label = "scripted controller" if args.policy == "scripted" else "RL policy (probe 5, it 250)"
    rec = {"close": [], "wide": [], "phase": []}
    if args.mode == "video":
        tmp = OUT / "_tmp"
        tmp.mkdir(exist_ok=True)
        w_close = [writer(OUT / f"{args.tag}closeup_env{i}.mp4") for i in range(N)]
        w_wide = [writer(tmp / f"wide_env{i}.mp4") for i in range(N)]
        w_grid = writer(OUT / f"{args.tag}closeup_grid_3x3.mp4") if N >= 9 else None
        gif = []
        phase_log = np.zeros((EPISODE_STEPS, N), np.int64)
    else:
        w_tiles = writer(Path(args.out))
    state = {"aimed": False}

    def pol(obs):
        if not state["aimed"]:
            obj, cmd_term = u.scene["object"], u.command_manager.get_term("object_pose")
            if args.mode == "frames" and args.policy == "scripted":
                Path(args.out).with_suffix(".init.json").write_text(json.dumps({
                    "cube_pose_w": torch.cat([_t(obj.data.root_pos_w), _t(obj.data.root_quat_w)], 1).tolist(),
                    "command_b": cmd_term.pose_command_b.tolist()}))
            if args.init:
                init = json.loads(Path(args.init).read_text())
                ids = torch.arange(N, device=u.device)
                obj.write_root_pose_to_sim_index(root_pose=torch.tensor(init["cube_pose_w"], device=u.device), env_ids=ids)
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=u.device), env_ids=ids)
                cmd_term.pose_command_b[:] = torch.tensor(init["command_b"], device=u.device)
                obs = run_env.get_observations()  # observation of the replayed start state
            aim_task(u, cam)
            if args.mode == "video":
                aim_wide(u, u.scene["wide_cam"])
            state["aimed"] = True
        return policy(obs)

    def on_step(t, uu):
        cube, goal = cube_and_target(uu)
        hit = (torch.linalg.norm(cube - goal, dim=-1) < th.success_r).cpu().numpy()
        tp = target_points(uu)
        fr = grab(uu, "video_cam")
        uvd = project(uu, "video_cam", tp)
        ph = ctrl.phase.cpu().numpy() if ctrl is not None else None
        frames = []
        for i in range(N):
            lines = ([f"phase {ph[i] + 1}/6: {PHASES[ph[i]].upper()}"] if ph is not None else ["RL policy"]) + \
                    [f"{label}   t = {(t + 1) * uu.step_dt:.2f} s"]
            frames.append(overlay(fr[i], uvd[i], lines, bool(hit[i])))
        if args.mode == "frames":
            w_tiles.append_data(tile(np.stack([f[::2, ::2] for f in frames[:4]]), 2))
            return
        phase_log[t] = ph
        for i in range(N):
            w_close[i].append_data(frames[i])
        fw = grab(uu, "wide_cam")
        uvw = project(uu, "wide_cam", tp)
        for i in range(N):
            w_wide[i].append_data(overlay(fw[i], uvw[i], [f"phase {ph[i] + 1}/6: {PHASES[ph[i]].upper()}",
                                                          f"{label}   t = {(t + 1) * uu.step_dt:.2f} s"], bool(hit[i])))
        if w_grid is not None:
            g = tile(np.stack([f[::2, ::2] for f in frames[:9]]), 3)
            w_grid.append_data(g)
            if t % 4 == 0:
                gif.append(g[::4, ::4])

    if ctrl is not None:
        ctrl.started = False
    res = rollout(run_env, pol, th, on_step=on_step)
    cube, goal = cube_and_target(u)
    per_env = [{"env": i, "cube_start_xy": None, "target": [round(float(x), 3) for x in goal[i]],
                **{k: (bool(res[k][i]) if res[k].dtype == torch.bool else round(float(res[k][i]), 4))
                   for k in ("success", "success_5cm", "clean", "lift", "slam", "jam", "dist")}} for i in range(N)]
    if args.mode == "frames":
        w_tiles.close()
        Path(args.out).with_suffix(".json").write_text(json.dumps({"policy": args.policy, "task": task, "label": label,
                                                                   "checkpoint": args.checkpoint if args.policy == "rl" else None,
                                                                   "per_env": per_env[:4]}, indent=1))
        env.close()
        sys.exit(0)

    for w in w_close + w_wide:
        w.close()
    if w_grid is not None:
        w_grid.close()
        imageio.mimsave(OUT / f"{args.tag}closeup_grid_3x3.gif", gif, duration=80, loop=0)
    import shutil

    succ = [i for i in range(N) if per_env[i]["success"] and per_env[i]["clean"]]
    single = succ[0] if succ else None
    audit = ROOT / "results" / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    phase_frames = {}
    if single is not None:
        shutil.copyfile(OUT / f"{args.tag}closeup_env{single}.mp4", OUT / f"{args.tag}single_success_closeup.mp4")
        shutil.copyfile(tmp / f"wide_env{single}.mp4", OUT / f"{args.tag}single_success_wide.mp4")
        # audit frames: read back from the success close-up (one frame per control step), half resolution
        rd = imageio.get_reader(str(OUT / f"{args.tag}closeup_env{single}.mp4"))
        frames_single = [f[::2, ::2] for f in rd]
        rd.close()
        for k, name in enumerate(PHASES):
            steps = np.flatnonzero(phase_log[:, single] == k)
            if len(steps) == 0:
                continue
            pick = sorted({int(steps[0]), int(steps[len(steps) // 2]), int(steps[-1])} |
                          ({int(steps[len(steps) // 4]), int(steps[3 * len(steps) // 4])} if len(steps) >= 5 else set()))
            pick = pick if len(pick) >= 3 else sorted(set(pick) | {int(s) for s in steps[:3]})
            sheet = np.concatenate([frames_single[s] for s in pick], axis=1)
            fname = f"scripted_demo_env{single}_phase{k + 1}_{name}_steps_{'-'.join(map(str, pick))}.png"
            imageio.imwrite(audit / fname, sheet)
            phase_frames[name] = {"steps": pick, "sheet": fname, "first_step": int(steps[0]), "last_step": int(steps[-1])}
    shutil.rmtree(tmp, ignore_errors=True)
    meta = {"task": TASK, "seed": seed, "num_envs": N, "per_env": per_env, "single_success_env": single,
            "phase_frames": phase_frames,
            "phase_entry_steps": {i: {PHASES[k]: int(np.argmax(phase_log[:, i] == k)) if (phase_log[:, i] == k).any() else None
                                      for k in range(6)} for i in range(N)}}
    (OUT / f"{args.tag}videos.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
