"""Task-space arm actions through the step-A DLS IK (section 17, run 3 onward).

The policy outputs 4 numbers in [-1, 1]: fingertip dx, dy, dz (x 1 cm per step) and dyaw (x 3 deg per step). They move
a fingertip target and a hand-yaw target; the hand is held vertical by the IK (the policy cannot tilt it). Each control
step the step-A tracking code (acceleration-limited setpoint, DLS IK, gravity/cube/acceleration/velocity feed-forward;
lift_rl/scripted.py) turns the target into the 7 arm joint-position targets. DEVIATION: arm actions are Cartesian deltas
via IK; joint targets computed by IK.
"""

from __future__ import annotations

import math

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz

from .metrics import Thresholds
from .scripted import ScriptedController

STEP_POS = 0.01  # m per step at action 1
STEP_YAW = math.radians(3.0)  # rad per step at action 1
LEAD_MAX = 0.05  # the target may lead the fingertip by at most 5 cm (no wind-up)
BOX_LO = (0.25, -0.45, 0.005)  # fingertip workspace box, env frame [m]
BOX_HI = (0.80, 0.45, 0.60)


def _t(x):
    return x.torch if hasattr(x, "torch") else x


class _IK(ScriptedController):
    """Per-env resettable tracking state around ScriptedController._track."""

    V_TRACK = 0.60  # setpoint speed cap toward the target [m/s] (1 cm/step = 0.5 m/s)

    def reset_ids(self, ids):
        if not self.started:
            self._reset()
        tcp = self._tcp()
        self.sp[ids] = tcp[ids]
        self.v[ids] = 0.0
        self.q_sp[ids] = _t(self.robot.data.body_link_quat_w)[ids, self.hand]


class TaskSpaceIKAction(ActionTerm):
    cfg: "TaskSpaceIKActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 4, device=self.device)
        self._q_cmd = torch.zeros(self.num_envs, 7, device=self.device)
        self.target = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw = torch.zeros(self.num_envs, device=self.device)
        self.ik = None
        self.lo = torch.tensor(BOX_LO, device=self.device)
        self.hi = torch.tensor(BOX_HI, device=self.device)
        self._need_init = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._scale, self._offset = 1.0, 0.0  # ScriptedController maps its output with these: joint targets as-is

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._q_cmd

    def _lazy(self):
        if self.ik is None:
            self.ik = _IK(self._env, Thresholds(), reflex=True)
            self.ik._reset()
            self.arm_ids = self.ik.arm_ids

    def _init_ids(self, ids):
        tcp = self.ik._tcp()
        R = matrix_from_quat(_t(self.ik.robot.data.body_link_quat_w)[:, self.ik.hand])
        self.target[ids] = tcp[ids]
        self.yaw[ids] = torch.atan2(R[ids, 1, 0], R[ids, 0, 0])
        self.ik.reset_ids(ids)

    def process_actions(self, actions: torch.Tensor):
        self._lazy()
        if self._need_init.any():
            ids = self._need_init.nonzero(as_tuple=False)[:, 0]
            self._init_ids(ids)
            self._need_init[ids] = False
        a = actions.clamp(-1.0, 1.0)
        self._raw[:] = a
        ik = self.ik
        p = ik._tcp()
        self.target = torch.maximum(torch.minimum(self.target + a[:, :3] * STEP_POS, self.hi), self.lo)
        lead = self.target - p
        n = torch.linalg.norm(lead, dim=-1, keepdim=True)
        self.target = p + lead * torch.clamp(LEAD_MAX / n.clamp_min(1e-9), max=1.0)
        self.yaw = self.yaw + a[:, 3] * STEP_YAW
        r = ik.robot
        qh = _t(r.data.body_link_quat_w)[:, ik.hand]
        R = matrix_from_quat(qh)
        z0 = torch.zeros_like(self.yaw)
        q_des = quat_from_euler_xyz(torch.full_like(self.yaw, math.pi), z0, self.yaw)  # vertical, policy yaw
        pm = getattr(self._env, "_phase_machine", None)
        held = pm.grasp if pm is not None else torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        vmax = torch.full((self.num_envs,), ik.V_TRACK, device=self.device)
        dn = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        a_arm = ik._track(self.target, vmax, dn, q_des, p, qh, R, held)
        self._q_cmd = a_arm * ik.a_scale + ik.a_offset

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._q_cmd, joint_ids=self.arm_ids)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._need_init[ids] = True
        self._raw[ids] = 0.0


@configclass
class TaskSpaceIKActionCfg(ActionTermCfg):
    class_type: type = TaskSpaceIKAction


def scripted_delta_policy(ctrl, term):
    """Drive the step-A controller through the 4-D task-space interface: its profiled fingertip setpoint and face-aligned
    yaw become clamped per-step deltas of the action term's targets."""
    import math as _m

    from isaaclab.utils.math import euler_xyz_from_quat

    def policy(obs):
        ctrl(obs)  # advances the controller's own phase machine and setpoint profile (its joint output is unused)
        r = ctrl.robot
        R = matrix_from_quat(_t(r.data.body_link_quat_w)[:, ctrl.hand])
        _, _, cube_yaw = euler_xyz_from_quat(_t(ctrl.obj.data.root_quat_w))
        psi_now = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        psi = cube_yaw + torch.round((psi_now - cube_yaw) / (_m.pi / 2)) * (_m.pi / 2)
        dyaw = torch.remainder(psi - term.yaw + _m.pi, 2 * _m.pi) - _m.pi
        dpos = (ctrl.sp - term.target) / STEP_POS
        return torch.cat([dpos, (dyaw / STEP_YAW)[:, None]], 1).clamp(-1.0, 1.0)

    return policy
