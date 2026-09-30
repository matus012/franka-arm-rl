"""Newton (MuJoCo-Warp) physics settings and the pre-run environment probe.

The stock lift task is PhysX-only (gotcha 4a.1). The Newton config below is copied from the
Franka *stack* task's ``newton_mjwarp`` preset
(``isaaclab_tasks/manager_based/manipulation/stack/stack_env_cfg.py``, BSD-3-Clause,
Copyright (c) 2022-2026, The Isaac Lab Project Developers).
"""

from __future__ import annotations

import importlib.util

import torch
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

# Legacy Franka USD: the default ISAACLAB_NUCLEUS_DIR/Robots/FrankaEmika/panda_instanceable.usd 404s (gotcha 4a.2).
FRANKA_LEGACY_USD_SUFFIX = "Robots/FrankaEmika/Legacy/panda_instanceable.usd"


@configclass
class ContactShapeCfg(NewtonShapeCfg):
    """NewtonShapeCfg + contact stiffness/damping (forwarded onto Newton's ShapeConfig).

    Newton maps (ke, kd) to MuJoCo solref = (2/kd, kd/2*sqrt(1/ke)). Upstream default
    (2500, 100) -> solref (0.02 s, 1.0).
    """

    ke: float = 2.5e3
    kd: float = 100.0


def newton_mjwarp_cfg(
    iterations: int = 100,
    ls_iterations: int = 15,
    num_substeps: int = 2,
    contact_ke: float | None = None,
    contact_kd: float | None = None,
) -> NewtonCfg:
    """Newton + MuJoCo-Warp physics, values from the stack task's ``newton_mjwarp`` preset.

    ``contact_ke``/``contact_kd`` = None keeps the preset's (upstream default) contact stiffness.
    """
    shape_cfg = NewtonShapeCfg()
    if contact_ke is not None or contact_kd is not None:
        shape_cfg = ContactShapeCfg(ke=contact_ke or 2.5e3, kd=contact_kd or 100.0)
    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=iterations,
            ls_iterations=ls_iterations,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=shape_cfg,
        num_substeps=num_substeps,
        debug_mode=False,
    )


def set_contact_solimp(env, env_ids, solimp: tuple[float, float, float, float, float]) -> None:
    """Startup event: set MuJoCo solimp (d0, dmax, width, midpoint, power) on every shape.

    Newton stores it as the per-shape custom attribute ``model.mujoco.geom_solimp`` (default
    (0.9, 0.95, 0.001, 0.5, 2)) and copies it into MuJoCo-Warp on a SHAPE_PROPERTIES notify.
    """
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverNotifyFlags

    attr = NewtonManager.get_model().mujoco.geom_solimp
    wp.to_torch(attr)[:] = torch.tensor(solimp, device=wp.to_torch(attr).device)
    NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)


def assert_scipy() -> None:
    """Without scipy Newton silently swaps convex hulls for bounding boxes (gotcha 4a.3)."""
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError("scipy is not importable: Newton would fall back to bounding-box collision hulls. STOP.")
    import scipy.spatial  # noqa: F401  (the hull code path)


def assert_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. A CPU fallback is a STOP, not a slow run.")


def assert_run_preconditions(env_cfg, device: str | None = None) -> None:
    """Probe before any GPU job: CUDA up, scipy importable, physics is NewtonCfg, device is cuda:0."""
    assert_cuda()
    assert_scipy()
    if not isinstance(env_cfg.sim.physics, NewtonCfg):
        raise RuntimeError(f"physics is {type(env_cfg.sim.physics).__name__}, expected NewtonCfg. STOP.")
    dev = device if device is not None else env_cfg.sim.device
    if str(dev) != "cuda:0":
        raise RuntimeError(f"sim device is {dev!r}, expected 'cuda:0'. STOP.")


def assert_on_cuda(env, policy_module: torch.nn.Module | None = None) -> None:
    """Post-construction check: the live env (and policy) really sit on cuda:0."""
    if str(env.unwrapped.device) != "cuda:0":
        raise RuntimeError(f"env device is {env.unwrapped.device}, expected cuda:0. STOP.")
    if policy_module is not None:
        p = next(policy_module.parameters())
        if p.device.type != "cuda":
            raise RuntimeError(f"policy parameters on {p.device}, expected cuda. STOP.")
