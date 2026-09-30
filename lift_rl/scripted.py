"""Step-A scripted controller (section 14, frozen copy of scripts/scripted_demo.py's ScriptedController) for the
section-16 fixed world: DLS IK + feed-forward -> joint-position arm actions.

  reflex=True : 7 arm actions only; the env's gripper reflex closes the claw. The controller descends to the grasp
                height, waits there until the env's phase machine confirms the grasp, then lifts, carries and holds.
  reflex=False: 8 actions (arm + binary gripper), as in section 14.
RakeController: the run-1 exploit for the proof (hand tilted 40 deg, lunge past the cube low over the table and
drag it back toward the robot), same IK.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    compute_pose_error, euler_xyz_from_quat, matrix_from_quat, quat_from_angle_axis, quat_from_euler_xyz, quat_mul)

from .evaluation import contact_readings, cube_and_target
from .metrics import CUBE_REST_Z, Thresholds
from .phases import CARRY, CLOSE, DESCEND, HOLD, HOVER, LIFT, PHASES
from .staged import yaw_mod90_error


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

    def __init__(self, env, th: Thresholds, reflex: bool = False, debug_env: int = -1):
        self.u = u = env.unwrapped
        self.reflex, self.debug_env = reflex, debug_env
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
        if self.reflex:  # wait for the env's phase machine to confirm the grasp (reflex squeeze)
            gm = getattr(u, "_phase_machine", None)
            go = gm.grasp if gm is not None else touched
        else:
            go = touched | (self.phase_t >= self.CLOSE_MAX_STEPS)
        m = self._advance((ph == CLOSE) & go, LIFT)
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
        held = (ph >= LIFT) & both
        a_arm = self._track(goal_pt, vmax, ph == DESCEND, q_des, p, qh, R, held)
        grip = torch.where(ph >= CLOSE, -1.0, 1.0)[:, None]

        if self.debug_env >= 0 and self.t % 5 == 0:
            i = self.debug_env
            print(f"[dbg] t={self.t:3d} ph={PHASES[int(ph[i])]:7s} p=({p[i,0]:.3f},{p[i,1]:.3f},{p[i,2]:.3f}) "
                  f"sp=({self.sp[i,0]:.3f},{self.sp[i,1]:.3f},{self.sp[i,2]:.3f}) cube=({cube[i,0]:.3f},{cube[i,1]:.3f},{cube[i,2]:.3f}) "
                  f"goal=({goal[i,0]:.3f},{goal[i,1]:.3f},{goal[i,2]:.3f}) xy={xy_err[i]*1000:.1f}mm yaw={math.degrees(yaw_err[i]):.1f} "
                  f"tilt={math.degrees(tilt[i]):.1f} lf={c['lf'][i]:.1f} rf={c['rf'][i]:.1f} palm={c['palm'][i]:.1f} "
                  f"tbl={c['robot_table'][i]:.1f}", flush=True)
        self.phase_t += 1
        self.t += 1
        return a_arm if self.reflex else torch.cat([a_arm, grip], 1)

    def _track(self, goal_pt, vmax, dn, q_des, p, qh, R, held):
        """Setpoint profile toward goal_pt (speed cap vmax [m/s], A_MAX; dn: vertical-descent rows keep xy on the
        goal), orientation toward q_des (W_MAX), DLS IK + feed-forward -> arm action (N,7)."""
        u, r = self.u, self.robot
        dt = u.step_dt
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
        return a_arm


class RakeController(ScriptedController):
    """The run-1 exploit: hand tilted 40 deg about y, approach 14 cm beyond the cube at 12 cm height, lunge down to
    1.2 cm above the table and drag the cube 13 cm back toward the robot; never vertical, so the reflex never closes."""

    def __call__(self, obs=None) -> torch.Tensor:
        if not self.started:
            self._reset()
            cube0, _ = cube_and_target(self.u)
            self.c0 = cube0.clone()
            self.c0[:, 2] = CUBE_REST_Z
        u, r = self.u, self.robot
        p = self._tcp()
        qh = _t(r.data.body_link_quat_w)[:, self.hand]
        R = matrix_from_quat(qh)
        V = lambda x, y, z: torch.tensor([x, y, z], device=self.dev)  # noqa: E731
        t = self.t
        if t < 60:
            goal = self.c0 + V(0.14, 0, 0.12)
        elif t < 100:
            goal = self.c0 + V(0.10, 0, 0.012 - CUBE_REST_Z)
        else:
            goal = self.c0 + V(-0.03, 0, 0.012 - CUBE_REST_Z)
        z0 = torch.zeros(self.N, device=self.dev)
        q_des = quat_from_euler_xyz(torch.full_like(z0, math.pi), torch.full_like(z0, math.radians(-40.0)), z0)
        vmax = torch.full((self.N,), 0.25, device=self.dev)
        a = self._track(goal, vmax, torch.zeros(self.N, dtype=torch.bool, device=self.dev), q_des, p, qh, R,
                        torch.zeros(self.N, dtype=torch.bool, device=self.dev))
        self.t += 1
        return a if self.reflex else torch.cat([a, torch.ones(self.N, 1, device=self.dev)], 1)
