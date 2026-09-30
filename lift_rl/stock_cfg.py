"""M0: the stock Isaac Lab Franka cube-lift task with only the 4a fixes applied.

Changes vs ``Isaac-Lift-Cube-Franka-v0`` (everything else is stock: reward, observations, actions,
events, curriculum, PPO config):
- 4a.1 physics: PhysX -> Newton / MuJoCo-Warp (stack task's ``newton_mjwarp`` preset),
- 4a.2 robot USD: Legacy panda path (the default one 404s),
- 4a.4 NaN guard: env class is :class:`lift_rl.guarded_env.GuardedRLEnv` (registration).
"""

from __future__ import annotations

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import (
    FrankaCubeLiftEnvCfg as FrankaCubeLiftIKAbsEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

from .physics import FRANKA_LEGACY_USD_SUFFIX, newton_mjwarp_cfg


def apply_newton_fixes(cfg) -> None:
    """The 4a.1 + 4a.2 fixes, shared by the stock (M0) and own (M1/M2) configs."""
    cfg.sim.physics = newton_mjwarp_cfg()
    cfg.scene.robot.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/{FRANKA_LEGACY_USD_SUFFIX}"


@configclass
class StockLiftNewtonEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_newton_fixes(self)


PHYSICS_FIX1 = {"finger_armature": 0.1, "contact_ke": 1.0e4, "contact_kd": 200.0}


@configclass
class StockLiftNewtonFixEnvCfg(StockLiftNewtonEnvCfg):
    """M0 physics-config fix attempt 1: finger armature 0.1 + contact solref 0.01 s (ke 1e4, kd 200).

    Reward, observations, actions, events, curriculum and PPO stay stock.
    """

    def __post_init__(self):
        super().__post_init__()
        f = PHYSICS_FIX1
        self.sim.physics = newton_mjwarp_cfg(contact_ke=f["contact_ke"], contact_kd=f["contact_kd"])
        self.scene.robot.actuators["panda_hand"].armature = f["finger_armature"]


PHYSICS_FIX2 = {"finger_armature": 0.1, "num_substeps": 4, "contact_ke": 4.0e4, "contact_kd": 400.0}


@configclass
class StockLiftNewtonFix2EnvCfg(StockLiftNewtonEnvCfg):
    """M0 physics-config fix attempt 2: fix 1 + 4 Newton substeps (solver dt 2.5 ms) + contact solref
    0.005 s (ke 4e4, kd 400; MuJoCo needs solref >= 2 solver dt). 4x stiffer contacts -> ~4x less
    penetration, which is what the fix-1 policy turned into cube launches (videos/m0_fix1).
    Reward, observations, actions, events, curriculum and PPO stay stock.
    """

    def __post_init__(self):
        super().__post_init__()
        f = PHYSICS_FIX2
        self.sim.physics = newton_mjwarp_cfg(num_substeps=f["num_substeps"], contact_ke=f["contact_ke"], contact_kd=f["contact_kd"])
        self.scene.robot.actuators["panda_hand"].armature = f["finger_armature"]


@configclass
class StockLiftIKNewtonEnvCfg(FrankaCubeLiftIKAbsEnvCfg):
    """Diagnostic only: the stock IK-abs lift variant (high-PD arm) on Newton, for the scripted grasp."""

    def __post_init__(self):
        super().__post_init__()
        apply_newton_fixes(self)
