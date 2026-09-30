"""MDP terms for the own lift env (M1/M2). Stock Isaac Lab terms are re-exported; new ones are defined here.

All functions return (num_envs,) tensors (rewards/terminations) or (num_envs, k) (observations).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import *  # noqa: F401,F403  (stock generic terms: joint_pos_rel, last_action, ...)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, euler_xyz_from_quat, quat_mul, quat_inv
from isaaclab_tasks.manager_based.manipulation.lift.mdp import (  # noqa: F401  stock lift terms
    object_ee_distance,
    object_goal_distance,
    object_is_lifted,
    object_position_in_robot_root_frame,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .metrics import CUBE_REST_Z

CUBE_HALF = 0.024  # DexCube at 0.8 scale: 4.8 cm edge


def _t(x):
    return x.torch if hasattr(x, "torch") else x


# -- observations ------------------------------------------------------------------------------
def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube orientation: quaternion (xyzw, sign-canonical w >= 0) + (sin 4*yaw, cos 4*yaw).

    The cube looks the same every 90 deg of yaw, so (sin 4y, cos 4y) is the grasp-relevant yaw
    without the 4-fold ambiguity. (N, 6).
    """
    robot, obj = env.scene[robot_cfg.name], env.scene[object_cfg.name]
    q = quat_mul(quat_inv(_t(robot.data.root_quat_w)), _t(obj.data.root_quat_w))
    q = torch.where(q[:, 3:4] < 0, -q, q)
    _, _, yaw = euler_xyz_from_quat(q)
    return torch.cat([q, torch.sin(4 * yaw)[:, None], torch.cos(4 * yaw)[:, None]], dim=1)


# -- rewards -----------------------------------------------------------------------------------
def finger_width(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    names = robot.data.joint_names
    q = _t(robot.data.joint_pos)
    return q[:, names.index("panda_finger_joint1")] + q[:, names.index("panda_finger_joint2")]


def is_grasped(
    env: ManagerBasedRLEnv,
    max_ee_dist: float = 0.03,
    width_range: tuple[float, float] = (2 * CUBE_HALF - 0.013, 2 * CUBE_HALF + 0.004),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Geometric grasp: fingertip center within 3 cm of the cube center AND the fingers stopped by the
    cube (finger gap ~ cube width, i.e. neither open at 8 cm nor closed on nothing). Float 0/1."""
    obj = env.scene[object_cfg.name]
    ee = _t(env.scene[ee_frame_cfg.name].data.target_pos_w)[..., 0, :]
    d = torch.linalg.norm(_t(obj.data.root_pos_w) - ee, dim=1)
    w = finger_width(env)
    return ((d < max_ee_dist) & (w > width_range[0]) & (w < width_range[1])).float()


def grasp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 while the cube is held between the fingertips (see is_grasped)."""
    return is_grasped(env)


def lift_reward(env: ManagerBasedRLEnv, minimal_height: float = 0.04) -> torch.Tensor:
    """1 while the cube center is above ``minimal_height`` (world z; resting center is ~0.021)."""
    return object_is_lifted(env, minimal_height)


def table_slam_penalty(
    env: ManagerBasedRLEnv, force_threshold: float = 20.0, sensor_name: str = "contact_hand_table"
) -> torch.Tensor:
    """M2: 1 when the largest hand/finger-vs-table contact force exceeds ``force_threshold`` [N]."""
    s = env.scene[sensor_name]
    f = _t(s.data.force_matrix_w).reshape(env.num_envs, -1, 3).norm(dim=-1).amax(dim=1)
    return (f > force_threshold).float()


def jam_penalty(
    env: ManagerBasedRLEnv, contact_n: float = 1.0, off_table_dz: float = 0.02, sensor_name: str = "contact_cube_hand"
) -> torch.Tensor:
    """M2: 1 on a jam step (same rule as the evaluator, metrics.EpisodeTracker.jam_step):
    cube touches the palm, or it is off the table and touched by the robot (hand, fingers or any arm
    link) but not by both fingers."""
    s = env.scene[sensor_name]
    fm = _t(s.data.force_matrix_w).reshape(env.num_envs, -1, 3).norm(dim=-1)
    idx = {n: i for i, n in enumerate(s.filter_object_names)}
    palm = fm[:, idx["panda_hand"]] > contact_n
    lf, rf = fm[:, idx["panda_leftfinger"]] > contact_n, fm[:, idx["panda_rightfinger"]] > contact_n
    links = [i for n, i in idx.items() if n.startswith("panda_link")]
    arm = fm[:, links].amax(dim=1) > contact_n if links else torch.zeros_like(palm)
    z = _t(env.scene["object"].data.root_pos_w)[:, 2] - env.scene.env_origins[:, 2]
    off = z > CUBE_REST_Z + off_table_dz
    return (palm | (off & (palm | lf | rf | arm) & ~(lf & rf))).float()


def target_position_w(env: ManagerBasedRLEnv, command_name: str = "object_pose") -> torch.Tensor:
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command(command_name)
    pos, _ = combine_frame_transforms(_t(robot.data.root_pos_w), _t(robot.data.root_quat_w), cmd[:, :3])
    return pos
