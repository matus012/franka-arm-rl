"""M1/M2: this repo's own Franka cube-lift environment (Newton physics).

Started as a copy of Isaac Lab's ``Isaac-Lift-Cube-Franka-v0`` (lift_env_cfg.py + franka/joint_pos_env_cfg.py,
BSD-3-Clause, Copyright (c) 2022-2026, The Isaac Lab Project Developers). Changes vs stock:
- physics: Newton / MuJoCo-Warp (+ the M0 physics fix, see PHYSICS_FIX), Legacy Franka USD path,
- cube yaw randomized over (-pi, pi) at reset; cube orientation observed,
- reward terms named reach / grasp / lift / track_coarse / track_fine / track_precise / action_rate / joint_vel,
- M2 (LiftTwistEnvCfg): table-slam and jam penalties + cube mass / grasp-friction randomization.
Action space is stock: 7 arm joint-position targets (scale 0.5, offset = default pose) + binary gripper
(open 0.04 / closed 0.0 m per finger), 50 Hz control (dt 0.01, decimation 2), 5 s = 250-step episodes.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab_tasks.manager_based.manipulation.lift.mdp import UniformPoseCommandCfg
from isaaclab.envs.mdp import BinaryJointPositionActionCfg, JointPositionActionCfg

from . import mdp
from .instrument import add_eval_sensors
from .physics import FRANKA_LEGACY_USD_SUFFIX, newton_mjwarp_cfg
from .randomization import add_cube_randomization

# M0 physics carried into M1/M2 = fix 2, the only config under which PPO learned a real grasp
# (lift 99.5 %, jam 0.4 %; development log, not published). Stock Newton contacts cannot pinch the cube.
from .stock_cfg import PHYSICS_FIX2 as PHYSICS_FIX  # noqa: E402


@configclass
class LiftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector", offset=OffsetCfg(pos=[0.0, 0.0, 0.1034])
            )
        ],
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[0, 0, 0, 1]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0, 0, 0.707, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]), spawn=GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))


@configclass
class CommandsCfg:
    object_pose = UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    arm_action = JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)
    gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    grasp = RewTerm(func=mdp.grasp_reward, weight=2.0)
    lift = RewTerm(func=mdp.lift_reward, params={"minimal_height": 0.04}, weight=15.0)
    track_coarse = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose", "success_threshold": 0.03},
        weight=16.0,
    )
    track_fine = RewTerm(func=mdp.object_goal_distance, params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"}, weight=5.0)
    track_precise = RewTerm(func=mdp.object_goal_distance, params={"std": 0.01, "minimal_height": 0.04, "command_name": "object_pose"}, weight=5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000})


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """M1: own env, own reward."""

    scene: LiftSceneCfg = LiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = newton_mjwarp_cfg(
            num_substeps=PHYSICS_FIX["num_substeps"], contact_ke=PHYSICS_FIX["contact_ke"], contact_kd=PHYSICS_FIX["contact_kd"]
        )
        self.scene.robot.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_LEGACY_USD_SUFFIX}"
        self.scene.robot.actuators["panda_hand"].armature = PHYSICS_FIX["finger_armature"]


# M2 twist settings (brief 5.M2)
SLAM_WEIGHT = -2.0
JAM_WEIGHT = -2.0
MASS_RANGE = (0.1, 0.5)  # kg, default 0.216
FRICTION_RANGE = (0.5, 1.25)  # grasp mu, default 1.0


@configclass
class LiftTwistEnvCfg(LiftEnvCfg):
    """M2: M1 + table-slam and jam penalties + cube mass / grasp-friction randomization."""

    def __post_init__(self):
        super().__post_init__()
        add_eval_sensors(self)  # the penalties read the same contact sensors the evaluator uses
        self.rewards.table_slam = RewTerm(func=mdp.table_slam_penalty, params={"force_threshold": 20.0}, weight=SLAM_WEIGHT)
        self.rewards.jam = RewTerm(func=mdp.jam_penalty, weight=JAM_WEIGHT)
        add_cube_randomization(self, MASS_RANGE, FRICTION_RANGE, mode="reset")
