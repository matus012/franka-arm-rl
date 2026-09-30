"""Print the contact parameters Newton hands to MuJoCo-Warp for the gripper, cube and table shapes.

Usage: uv run python scripts/inspect_contacts.py [--task LiftRL-Stock-Newton-v0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

p = argparse.ArgumentParser()
p.add_argument("--task", default="LiftRL-Stock-Newton-v0")
args = p.parse_args()

env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 2
env_cfg.sim.device = "cuda:0"
with launch_simulation(env_cfg, {"headless": True}):
    env = gym.make(args.task, cfg=env_cfg)
    from isaaclab_newton.physics import NewtonManager as M

    m, s = M._model, M._solver
    labels = m.shape_label
    keep = [i for i, l in enumerate(labels) if "env_0/" in l and ("collision" in l.lower() or "Table" in l)]
    arrs = {k: getattr(m, k).numpy() for k in ("shape_material_mu", "shape_material_ke", "shape_material_kd", "shape_margin", "shape_gap", "shape_type", "shape_scale") if hasattr(m, k)}
    for i in keep:
        print(labels[i], {k: (v[i].tolist() if hasattr(v[i], "tolist") else v[i]) for k, v in arrs.items()})
    bl = m.body_label
    mass = m.body_mass.numpy()
    for i, l in enumerate(bl):
        if "env_0/" in l:
            print("mass", l, float(mass[i]))
    mm = s.mjw_model
    print("opt", {k: getattr(mm.opt, k) for k in ("impratio", "cone", "iterations", "ls_iterations") if hasattr(mm.opt, k)})
    for k in ("geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_condim", "geom_priority"):
        if hasattr(mm, k):
            a = getattr(mm, k).numpy()
            print(k, a.shape, a.reshape(-1, a.shape[-1])[:40].tolist() if a.ndim > 1 else a[:40].tolist())
    env.close()
