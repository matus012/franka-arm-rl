"""LiftRL-Staged-v0 (brief 10.2): the own env (fix-2 physics, cube yaw +-pi, cube-orientation observation)
with the staged pick reward from lift_rl/staged.py, always-on shaping, and table hit = episode end.

Reward terms (weight x value, Isaac Lab multiplies by dt = 0.02 s):
  staged      1.0  x (stage + 0.9 progress)       in [0, 4.9]
  hold_bonus  1.0  x [stage 4 and cube within 3 cm of target]
  action_rate -0.005 x |a_t - a_{t-1}|^2          from step 0 (no curriculum)   [probe 1: -0.01]
  joint_vel   -0.0002 x |qdot|^2                   from step 0                   [probe 1: -0.001]
  low_fast    -2.0 x max(0, hand speed - 0.25 m/s) while fingertips < 10 cm above the table
  table_hit   -50  x [any robot body vs table > 20 N]   (and the episode terminates)
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat

from isaaclab.envs.mdp import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction

from . import mdp
from .env_cfg import LiftEnvCfg
from .instrument import add_eval_sensors
from .randomization import add_cube_randomization
from .staged import StagedParams, low_fast_penalty, staged_reward

TABLE_HIT_N = 20.0  # same threshold as the eval's slam / clean-grasp detector


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def robot_table_force(env) -> torch.Tensor:
    s = env.scene["contact_hand_table"]  # every robot body vs the table shape
    return _t(s.data.force_matrix_w).reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=1)


def table_hit(env, threshold: float = TABLE_HIT_N) -> torch.Tensor:
    """Termination + penalty: any hand/finger/arm contact with the table above ``threshold`` [N]."""
    return robot_table_force(env) > threshold


class StagedState:
    """Per-step staged quantities, computed once per control step and shared by the reward terms."""

    def __init__(self, env, params: StagedParams):
        self.env, self.p = env, params
        self.prev_ee = torch.zeros(env.num_envs, 3, device=env.device)
        self.descended = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.step = -1
        self.out: dict[str, torch.Tensor] = {}

    def get(self) -> dict[str, torch.Tensor]:
        env = self.env
        if self.step == env.common_step_counter:
            return self.out
        origins = env.scene.env_origins
        ee = _t(env.scene["ee_frame"].data.target_pos_w)[..., 0, :] - origins
        fresh = env.episode_length_buf <= 1  # first step after a reset: no valid previous position
        speed = torch.linalg.norm(ee - self.prev_ee, dim=-1) / env.step_dt
        speed = torch.where(fresh, torch.zeros_like(speed), speed)
        self.prev_ee = ee.clone()
        robot, obj = env.scene["robot"], env.scene["object"]
        rot = matrix_from_quat(_t(robot.data.body_quat_w)[:, robot.body_names.index("panda_hand")])
        tilt = torch.arccos((-rot[:, 2, 2]).clamp(-1.0, 1.0))
        _, _, cube_yaw = euler_xyz_from_quat(_t(obj.data.root_quat_w))
        from .staged import yaw_mod90_error

        yaw_err = yaw_mod90_error(rot[:, :, 1], cube_yaw)
        cube = _t(obj.data.root_pos_w) - origins
        target = mdp.target_position_w(env) - origins
        s = env.scene["contact_cube_hand"]
        fm = _t(s.data.force_matrix_w).reshape(env.num_envs, -1, 3).norm(dim=-1)
        idx = {n: i for i, n in enumerate(s.filter_object_names)}
        g_term = env.action_manager.get_term("gripper_action")
        grip = g_term.closed_cmd if hasattr(g_term, "closed_cmd") else g_term.raw_actions[:, 0] < 0.0
        self.descended &= ~fresh  # new episode: no descent yet
        out = staged_reward(ee, speed, tilt, yaw_err, cube, target, fm[:, idx["panda_leftfinger"]],
                            fm[:, idx["panda_rightfinger"]], fm[:, idx["panda_hand"]], grip, self.p, self.descended)
        self.descended = out["descended"]
        out["ee"], out["speed"] = ee, speed
        log = env.extras.setdefault("log", {})
        for k in range(5):
            log[f"Metrics/stage_{k}_frac"] = float((out["stage"] == k).float().mean())
        log["Metrics/hand_speed_mean"] = float(speed.mean())
        log["Metrics/descended_frac"] = float(self.descended.float().mean())
        self.out, self.step = out, env.common_step_counter
        return out


def _state(env) -> StagedState:
    if not hasattr(env, "_staged_state"):
        env._staged_state = StagedState(env, StagedParams())
    return env._staged_state


class staged_pick(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env):
        super().__init__(cfg, env)

    def __call__(self, env) -> torch.Tensor:
        return _state(env).get()["reward"]


def hold_bonus(env) -> torch.Tensor:
    return _state(env).get()["bonus"].float()


def low_fast(env) -> torch.Tensor:
    st = _state(env).get()
    return low_fast_penalty(st["ee"][:, 2], st["speed"], _state(env).p)


def table_hit_penalty(env) -> torch.Tensor:
    return table_hit(env).float()


@configclass
class StagedRewardsCfg:
    staged = RewTerm(func=staged_pick, weight=1.0)
    hold_bonus = RewTerm(func=hold_bonus, weight=1.0)
    # probe 2: penalties scaled down (probe 1 froze; joint_vel was up to -0.24/step vs ~0.3 stage-0 reward)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002, params={"asset_cfg": SceneEntityCfg("robot")})
    low_fast = RewTerm(func=low_fast, weight=-2.0)
    table_hit = RewTerm(func=table_hit_penalty, weight=-50.0)


@configclass
class LiftStagedEnvCfg(LiftEnvCfg):
    """Run 2 main env."""

    rewards: StagedRewardsCfg = StagedRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        add_eval_sensors(self)  # the reward reads pad/palm/table contacts
        self.curriculum = None  # penalties on from step 0
        self.terminations.table_hit = DoneTerm(func=table_hit, params={"threshold": TABLE_HIT_N})


class AutoCloseGripperAction(BinaryJointPositionAction):
    """Probe 5 (section 11): the gripper closes automatically once the descent latch holds and the fingertips are
    in the grasp band (h >= 0.9) over a cube resting on the table. The trigger latches: the gripper stays closed
    until the descent latch is lost (the hand leaves the cube without holding it) or the episode resets; outside
    the latch the policy's gripper command applies. Arm motion is always the policy's. Uses the staged state of
    the previous control step (one-step latency). DEVIATION: gripper close is a scripted trigger, not learned."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.auto = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.closed_cmd = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        st = getattr(self._env, "_staged_state", None)
        if st is not None and st.out:
            o = st.out
            valid = self._env.episode_length_buf >= 1  # the cached staged state belongs to this episode
            self.auto |= valid & o["descended"] & o["at_bottom"] & o["over"] & o["on_table"]
            self.auto &= o["descended"] | ~valid
        self.closed_cmd = self.auto | (actions[:, 0] < 0.0)
        self._processed_actions = torch.where(self.closed_cmd[:, None], self._close_command, self._processed_actions)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self.auto[slice(None) if env_ids is None else env_ids] = False


@configclass
class AutoCloseGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = AutoCloseGripperAction


@configclass
class LiftStagedAutoCloseEnvCfg(LiftStagedEnvCfg):
    """Probe 5: staged reward + scripted close trigger (see AutoCloseGripperAction)."""

    def __post_init__(self):
        super().__post_init__()
        g = self.actions.gripper_action
        self.actions.gripper_action = AutoCloseGripperActionCfg(
            asset_name=g.asset_name, joint_names=g.joint_names,
            open_command_expr=g.open_command_expr, close_command_expr=g.close_command_expr)


@configclass
class LiftStagedDREnvCfg(LiftStagedEnvCfg):
    """Run 2 twist arm: staged reward + cube mass (0.1-0.5 kg) and grasp-friction (0.5-1.25) randomization."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_randomization(self, (0.1, 0.5), (0.5, 1.25), mode="reset")


@configclass
class LiftStagedAutoCloseDREnvCfg(LiftStagedAutoCloseEnvCfg):
    """Section 12 twist arm: the frozen probe-5 setup + cube mass (0.1-0.5 kg) and grasp-friction (0.5-1.25)
    randomization per reset."""

    def __post_init__(self):
        super().__post_init__()
        add_cube_randomization(self, (0.1, 0.5), (0.5, 1.25), mode="reset")
