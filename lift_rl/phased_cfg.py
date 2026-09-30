"""Section 16 (step 1): LiftRL-Phased-Fixed-v0. One fixed world, the shared phase machine (lift_rl/phases.py), a
gripper reflex instead of a gripper action, observations that show everything the reward uses, no terminations
except time-out and the cube falling off the table.

World: cube at (0.50, 0.00), yaw 0, default mass and friction; target fixed at (0.50, 0.15, 0.35); robot at its
default pose, gripper open; 8 s episodes (400 control steps at 50 Hz). Fix-2 physics (as every run-2 env).

The arm is learned; the claw is a reflex: close in the sweet spot until pressure, then hold.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat
from isaaclab.envs.mdp import JointPositionActionCfg

from . import mdp
from .env_cfg import LiftEnvCfg
from .instrument import add_eval_sensors
from .phases import PhaseMachine, PhaseParams, phase_one_hot
from .randomization import add_cube_randomization
from .staged import yaw_mod90_error

TCP_OFFSET = 0.1034  # fingertip center on the hand z-axis (same point as the ee_frame sensor)
CUBE_XY = (0.50, 0.00)
TARGET = (0.50, 0.15, 0.35)
EPISODE_S = 8.0
PAD_OBS_MAX_N = 20.0
SQUEEZE_M = 0.003  # reflex hold: finger width target = width at pressure - 3 mm


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# --------------------------------------------------------------------------- sim state -> machine inputs
def tcp_and_rot(env):
    """Fingertip center (env frame) and hand rotation matrix, from the hand link pose (fresh also on reset steps)."""
    r = env.scene["robot"]
    h = r.body_names.index("panda_hand")
    R = matrix_from_quat(_t(r.data.body_link_quat_w)[:, h])
    tcp = _t(r.data.body_link_pos_w)[:, h] + R[:, :, 2] * TCP_OFFSET - env.scene.env_origins
    return tcp, R


def cube_state(env):
    obj = env.scene["object"]
    cube = _t(obj.data.root_pos_w) - env.scene.env_origins
    _, _, yaw = euler_xyz_from_quat(_t(obj.data.root_quat_w))
    return cube, yaw


def contacts(env):
    s = env.scene["contact_cube_hand"]
    fm = _t(s.data.force_matrix_w).reshape(env.num_envs, -1, 3).norm(dim=-1)
    idx = {n: i for i, n in enumerate(s.filter_object_names)}
    table = _t(env.scene["contact_hand_table"].data.force_matrix_w).reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=1)
    return fm[:, idx["panda_leftfinger"]], fm[:, idx["panda_rightfinger"]], fm[:, idx["panda_hand"]], table


class PhaseState:
    """Owns the PhaseMachine; updates it once per control step (from the reward manager) and caches the output."""

    def __init__(self, env, params: PhaseParams):
        self.env = env
        self.m = PhaseMachine(env.num_envs, env.device, params)
        self.prev_tcp = torch.zeros(env.num_envs, 3, device=env.device)
        self.step_id = -1
        self.speed = torch.zeros(env.num_envs, device=env.device)

    def get(self) -> dict[str, torch.Tensor]:
        env = self.env
        if self.step_id == env.common_step_counter:
            return self.m.out
        tcp, R = tcp_and_rot(env)
        fresh = env.episode_length_buf <= 1
        speed = torch.linalg.norm(tcp - self.prev_tcp, dim=-1) / env.step_dt
        self.speed = torch.where(fresh, torch.zeros_like(speed), speed)
        self.prev_tcp = tcp.clone()
        tilt = torch.arccos((-R[:, 2, 2]).clamp(-1.0, 1.0))
        cube, yaw = cube_state(env)
        target = mdp.target_position_w(env) - env.scene.env_origins
        lf, rf, palm, table = contacts(env)
        out = self.m.step(tcp, self.speed, tilt, yaw_mod90_error(R[:, :, 1], yaw), cube, target, lf, rf, palm, table)
        out["tcp"] = tcp
        log = env.extras.setdefault("log", {})
        for k in range(6):
            log[f"Metrics/phase_{k + 1}_frac"] = float((out["phase"] == k).float().mean())
            log[f"Metrics/reached_{k + 1}"] = float(self.m.entered[:, k].float().mean())
        log["Metrics/cube_target_dist_median"] = float(out["dist"].median())
        carry = out["phase"] >= 4
        log["Metrics/cube_target_dist_median_carry_hold"] = float(out["dist"][carry].median()) if bool(carry.any()) else -1.0
        self.step_id = env.common_step_counter
        return out


def phase_state(env) -> PhaseState:
    if not hasattr(env, "_phase_state"):
        env._phase_state = PhaseState(env, PhaseParams())
        env._phase_machine = env._phase_state.m
    return env._phase_state


def reset_phase(env, env_ids):
    ps = phase_state(env)
    ps.m.reset(env_ids)
    tcp, _ = tcp_and_rot(env)
    ps.prev_tcp[env_ids] = tcp[env_ids]


# --------------------------------------------------------------------------- gripper reflex (no policy action)
class ReflexGripperAction(BinaryJointPositionAction):
    """16.2: open at spawn and in HOVER/DESCEND; from CLOSE on it closes (target 0) until both pads > 5 N, then holds a
    fixed squeeze: finger width target = width at that moment - 3 mm. Opens again on a reopen or a drop. Uses the phase
    machine's state from the previous control step (one-step latency). action_dim = 0: the policy has 7 arm actions."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self.squeezed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hold_target = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor):
        m = phase_state(self._env).m
        closing = m.phase >= 2  # CLOSE, LIFT, CARRY, HOLD
        self.squeezed &= closing
        pos = _t(self._asset.data.joint_pos)[:, self._joint_ids]
        latch = closing & ~self.squeezed & (m.lf > m.p.pad_confirm_n) & (m.rf > m.p.pad_confirm_n)
        self.hold_target = torch.where(latch[:, None], (pos - SQUEEZE_M / self._num_joints).clamp_min(0.0), self.hold_target)
        self.squeezed |= latch
        close = torch.where(self.squeezed[:, None], self.hold_target, self._close_command.expand_as(pos))
        self._processed_actions = torch.where(closing[:, None], close, self._open_command.expand_as(pos))

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.squeezed[ids] = False


@configclass
class ReflexGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = ReflexGripperAction


# --------------------------------------------------------------------------- observations (16.3)
def obs_tcp(env):
    return tcp_and_rot(env)[0]


def obs_tcp_to_cube(env):
    return cube_state(env)[0] - tcp_and_rot(env)[0]


def obs_cube_to_target(env):
    return mdp.target_position_w(env) - env.scene.env_origins - cube_state(env)[0]


def obs_target(env):
    return mdp.target_position_w(env) - env.scene.env_origins


def obs_orientation(env):
    """cos(tilt), sin/cos of 4x (hand closing-axis yaw - cube yaw)."""
    _, R = tcp_and_rot(env)
    _, yaw = cube_state(env)
    psi = torch.atan2(R[:, 1, 1], R[:, 0, 1])
    return torch.stack([-R[:, 2, 2], torch.sin(4 * (psi - yaw)), torch.cos(4 * (psi - yaw))], 1)


def obs_gripper(env):
    """finger width [m], left/right pad force / 20 N (clipped), palm-contact flag."""
    r = env.scene["robot"]
    ids = r.find_joints("panda_finger.*")[0]
    width = _t(r.data.joint_pos)[:, ids].sum(1, keepdim=True)
    lf, rf, palm, _ = contacts(env)
    return torch.cat([width, (lf / PAD_OBS_MAX_N).clamp(0, 1)[:, None], (rf / PAD_OBS_MAX_N).clamp(0, 1)[:, None],
                      (palm > 1.0).float()[:, None]], 1)


def obs_phase(env):
    m = phase_state(env).m
    return torch.cat([phase_one_hot(m.phase), m.grasp.float()[:, None]], 1)


@configclass
class PhasedObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_arm_action = ObsTerm(func=mdp.last_action, params={"action_name": "arm_action"})
        tcp = ObsTerm(func=obs_tcp)
        tcp_to_cube = ObsTerm(func=obs_tcp_to_cube)
        cube_to_target = ObsTerm(func=obs_cube_to_target)
        target = ObsTerm(func=obs_target)
        orientation = ObsTerm(func=obs_orientation)
        gripper = ObsTerm(func=obs_gripper)
        phase = ObsTerm(func=obs_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# --------------------------------------------------------------------------- rewards (16.4), stage units per step
def _r(key):
    def f(env):
        return phase_state(env).get()[key]

    f.__name__ = f"phase_{key}"
    return f


@configclass
class PhasedRewardsCfg:
    phase_level = RewTerm(func=_r("phase_level"), weight=1.0)  # phase number (1-6) + 0.9 x progress
    entry_bonus = RewTerm(func=_r("entry"), weight=2.0)  # first entry into phases 2-6 per episode
    hold_bonus = RewTerm(func=_r("hold"), weight=1.0)  # HOLD and cube within 1 cm of the target
    table = RewTerm(func=_r("table"), weight=-3.0)  # any robot part vs table > 20 N, no termination
    palm = RewTerm(func=_r("palm"), weight=-1.0)
    push = RewTerm(func=_r("push"), weight=-1.0)  # cube > 1.5 cm from spawn before the first confirmed grasp
    drop = RewTerm(func=_r("drop"), weight=-5.0)
    low_fast = RewTerm(func=_r("low_fast"), weight=-2.0)
    near_fast = RewTerm(func=_r("near_fast"), weight=-2.0)  # 17.3c: speed > 0.10 m/s with fingertips < 5 cm over the cube top
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)  # probe-5 weights
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class PhasedTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(func=mdp.root_height_below_minimum,
                               params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})


@configclass
class PhasedActionsCfg:
    arm_action = JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)
    gripper_action = ReflexGripperActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0})


@configclass
class LiftPhasedFixedEnvCfg(LiftEnvCfg):
    observations: PhasedObservationsCfg = PhasedObservationsCfg()
    actions: PhasedActionsCfg = PhasedActionsCfg()
    rewards: PhasedRewardsCfg = PhasedRewardsCfg()
    terminations: PhasedTerminationsCfg = PhasedTerminationsCfg()
    eval_steps: int = 400  # evaluation.rollout: one 8 s episode

    def __post_init__(self):
        super().__post_init__()
        add_eval_sensors(self)
        self.curriculum = None
        self.episode_length_s = EPISODE_S
        ev = self.events.reset_object_pose.params
        ev["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.scene.object.init_state.pos = (CUBE_XY[0], CUBE_XY[1], 0.055)
        self.scene.object.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        r = self.commands.object_pose.ranges
        r.pos_x, r.pos_y, r.pos_z = (TARGET[0],) * 2, (TARGET[1],) * 2, (TARGET[2],) * 2
        self.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        self.events.reset_phase = EventTerm(func=reset_phase, mode="reset")


# --------------------------------------------------------------------------- section 17 chain: one change per phase
@configclass
class LiftPhasedP3EnvCfg(LiftPhasedFixedEnvCfg):
    """Phase 3: cube position x +-0.10 m, y +-0.25 m around (0.50, 0.00); yaw 0, target fixed."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_pose.params["pose_range"] = {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0),
                                                              "yaw": (0.0, 0.0)}


@configclass
class LiftPhasedP4EnvCfg(LiftPhasedP3EnvCfg):
    """Phase 4: + cube yaw +-pi."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_object_pose.params["pose_range"]["yaw"] = (-math.pi, math.pi)


@configclass
class LiftPhasedP5EnvCfg(LiftPhasedP4EnvCfg):
    """Phase 5: + target x 0.40-0.60, y +-0.25, z 0.25-0.50 (the stock ranges)."""

    def __post_init__(self):
        super().__post_init__()
        r = self.commands.object_pose.ranges
        r.pos_x, r.pos_y, r.pos_z = (0.4, 0.6), (-0.25, 0.25), (0.25, 0.5)


@configclass
class LiftPhasedP6EnvCfg(LiftPhasedP5EnvCfg):
    """Phase 6: + twist, cube mass 0.1-0.5 kg and grasp friction 0.5-1.25 per reset."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_randomization(self, (0.1, 0.5), (0.5, 1.25), mode="reset")


# --------------------------------------------------------------------------- run-3 contingency: task-space actions
@configclass
class LiftPhasedFixedTSEnvCfg(LiftPhasedFixedEnvCfg):
    """Fixed world with 4-D task-space arm actions (fingertip dx, dy, dz <= 1 cm/step, dyaw <= 3 deg/step) through the
    step-A DLS IK; the hand is held vertical by the IK. DEVIATION: arm actions are Cartesian deltas via IK."""

    def __post_init__(self):
        super().__post_init__()
        from .task_space import TaskSpaceIKActionCfg

        self.actions.arm_action = TaskSpaceIKActionCfg(asset_name="robot")


def _ts(cfg):
    from .task_space import TaskSpaceIKActionCfg

    cfg.actions.arm_action = TaskSpaceIKActionCfg(asset_name="robot")


@configclass
class LiftPhasedP3TSEnvCfg(LiftPhasedP3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _ts(self)


@configclass
class LiftPhasedP4TSEnvCfg(LiftPhasedP4EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _ts(self)


@configclass
class LiftPhasedP5TSEnvCfg(LiftPhasedP5EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _ts(self)


@configclass
class LiftPhasedP6TSEnvCfg(LiftPhasedP6EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _ts(self)


# --------------------------------------------------------------------------- 17.3d: demonstration-state resets
DEMO_STATES = str(Path(__file__).resolve().parents[1] / "data" / "demo_states_fixed_ts.pt")


@configclass
class LiftPhasedFixedTSDemoEnvCfg(LiftPhasedFixedTSEnvCfg):
    """Fixed world, task-space actions, and 50 % of resets start from a recorded scripted state (training only;
    evaluation.make_eval_cfg removes the event, so evaluation always starts from the normal start pose)."""

    def __post_init__(self):
        super().__post_init__()
        from .demo_reset import restore_demo_states

        self.events.demo_reset = EventTerm(func=restore_demo_states, mode="reset",
                                           params={"path": DEMO_STATES, "fraction": 0.5})



def _demo(cfg, path: str):
    from .demo_reset import restore_demo_states

    cfg.events.demo_reset = EventTerm(func=restore_demo_states, mode="reset", params={"path": path, "fraction": 0.5})


def _demo_path(k: int) -> str:
    return str(Path(__file__).resolve().parents[1] / "data" / f"demo_states_p{k}_ts.pt")


@configclass
class LiftPhasedP3TSDemoEnvCfg(LiftPhasedP3TSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _demo(self, _demo_path(3))


@configclass
class LiftPhasedP4TSDemoEnvCfg(LiftPhasedP4TSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _demo(self, _demo_path(4))


@configclass
class LiftPhasedP5TSDemoEnvCfg(LiftPhasedP5TSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _demo(self, _demo_path(5))


@configclass
class LiftPhasedP6TSDemoEnvCfg(LiftPhasedP6TSEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _demo(self, _demo_path(6))
