"""Cube mass / grasp-friction randomization (M2) and fixed values for the robustness grid.

Friction: MuJoCo-Warp resolves a contact's friction as max(mu_geom1, mu_geom2) (equal priority), and the
fingers carry mu = 1.0. Lowering only the cube's mu would therefore change nothing at the grasp. The
"grasp friction" knob sets the SAME mu on the cube and on both finger shapes, so the finger-cube contact
friction equals the sampled value. The table keeps mu = 1.0 (cube-table friction stays max(mu, 1)).
Newton has one friction coefficient per shape (gotcha 4a.7).

Mass: the stock ``randomize_rigid_body_mass`` term ("abs", inertia rescaled). Default cube mass 0.216 kg.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

CUBE_MASS_DEFAULT = 0.216  # kg (DexCube at 0.8 scale, measured from the Newton model)
FINGER_BODIES = ("panda_leftfinger", "panda_rightfinger")


class randomize_grasp_friction(ManagerTermBase):
    """Sample mu ~ U(friction_range) per env; write it to the cube shape and both finger shapes."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        from isaaclab_newton.physics import NewtonManager
        from newton.solvers import SolverNotifyFlags

        self._mgr = NewtonManager
        self._flag = SolverNotifyFlags.SHAPE_PROPERTIES
        model = NewtonManager.get_model()
        robot, cube = env.scene["robot"], env.scene["object"]
        self._cube_mu = wp.to_torch(cube._root_view.get_attribute("shape_material_mu", model)[:, 0])
        self._robot_mu = wp.to_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        per_body = robot.num_shapes_per_body
        idx = []
        for name in FINGER_BODIES:
            b = robot.body_names.index(name)
            s = sum(per_body[:b])
            idx.extend(range(s, s + per_body[b]))
        self._finger_idx = torch.tensor(idx, dtype=torch.long, device=env.device)

    def __call__(self, env, env_ids: torch.Tensor | None, friction_range: tuple[float, float]):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = env_ids.to(env.device).long()
        lo, hi = friction_range
        mu = lo + (hi - lo) * torch.rand(len(env_ids), device=env.device)
        self._cube_mu[env_ids] = mu[:, None].expand(-1, self._cube_mu.shape[1])
        self._robot_mu[env_ids[:, None], self._finger_idx[None, :]] = mu[:, None]
        self._mgr.add_model_change(self._flag)


def add_cube_randomization(
    cfg, mass_range: tuple[float, float] | None, friction_range: tuple[float, float] | None, mode: str = "reset"
) -> None:
    """Attach mass and/or grasp-friction events to an env cfg."""
    if mass_range is not None:
        cfg.events.cube_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode=mode,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "mass_distribution_params": mass_range,
                "operation": "abs",
                "recompute_inertia": True,
            },
        )
    if friction_range is not None:
        cfg.events.grasp_friction = EventTermCfg(
            func=randomize_grasp_friction, mode=mode, params={"friction_range": friction_range}
        )


def set_fixed_cube_physics(cfg, mass: float | None, friction: float | None) -> None:
    """Robustness grid cell: fixed cube mass and grasp friction for every episode (replaces any DR)."""
    for name in ("cube_mass", "grasp_friction"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    add_cube_randomization(
        cfg,
        None if mass is None else (mass, mass),
        None if friction is None else (friction, friction),
        mode="startup",
    )
