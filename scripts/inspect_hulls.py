"""World-space AABBs of the hand/finger/cube collision shapes vs body origins (env 0, default pose).

Usage: uv run python scripts/inspect_hulls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np

import lift_rl  # noqa: F401
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

TASK = "LiftRL-Stock-Newton-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.sim.device = "cuda:0"


def quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:  # q = xyzw
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return v @ R.T


with launch_simulation(env_cfg, {"headless": True}):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    from isaaclab_newton.physics import NewtonManager as M

    m, st = M._model, M._state_0
    body_q = st.body_q.numpy()
    shape_body = m.shape_body.numpy()
    shape_tf = m.shape_transform.numpy()
    scale = m.shape_scale.numpy()
    for i, lab in enumerate(m.shape_label):
        if not any(k in lab for k in ("panda_hand/coll", "finger/coll", "Object/coll", "link7/coll")):
            continue
        src = m.shape_source[i]
        b = shape_body[i]
        bq = body_q[b]
        if src is not None and hasattr(src, "vertices"):
            v = np.asarray(src.vertices) * scale[i]
        else:  # box half extents
            h = scale[i]
            v = np.array([[sx, sy, sz] for sx in (-h[0], h[0]) for sy in (-h[1], h[1]) for sz in (-h[2], h[2])])
        v = quat_rot(shape_tf[i][3:], v) + shape_tf[i][:3]
        w = quat_rot(bq[3:], v) + bq[:3]
        if "panda_hand/coll" in lab:
            print("hand hull extents (hand body frame) min", np.round(v.min(0), 4), "max", np.round(v.max(0), 4))
            zs = v[:, 2]
            print("hand hull verts with z > 0.05:", np.round(v[zs > 0.05], 4).tolist())
        if "rightfinger" in lab:  # hull vertices in the finger body frame (hand z = approach axis)
            print("rightfinger hull verts (body frame, sorted by z):")
            for row in sorted(np.round(v, 4).tolist(), key=lambda r: r[2]):
                print("   ", row)
        print(f"{lab.split('env_0/')[-1]:45s} nverts={len(v):5d} body_origin={np.round(bq[:3], 4)} "
              f"aabb_min={np.round(w.min(0), 4)} aabb_max={np.round(w.max(0), 4)}")
    env.close()
