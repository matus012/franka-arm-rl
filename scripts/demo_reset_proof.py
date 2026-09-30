"""17.3d proof: every env restored from a random demo snapshot (fraction 1.0), zero task-space action (hold still) for
50 steps (1 s). Per snapshot phase: the phase machine resumes in the restored phase (checked after the first step; a
forward step by the phase rules is reported separately), and for grasped snapshots (LIFT/CARRY/HOLD) the cube is still
held by both pads after 1 s with the fingertip within 1 cm of where it started.
Writes results/demo_reset_proof.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.demo_reset import restore_demo_states  # noqa: E402
from lift_rl.evaluation import make_eval_cfg  # noqa: E402
from lift_rl.phased_cfg import DEMO_STATES, contacts, cube_state, tcp_and_rot  # noqa: E402
from lift_rl.phases import LIFT, PHASES  # noqa: E402
from lift_rl.physics import assert_run_preconditions  # noqa: E402

TASK = "LiftRL-Phased-Fixed-TS-Demo-v0"
N = 256
cfg = make_eval_cfg(load_cfg_from_registry(TASK, "env_cfg_entry_point"), N, 5)
cfg.events.demo_reset = EventTermCfg(func=restore_demo_states, mode="reset", params={"path": DEMO_STATES, "fraction": 1.0})
assert_run_preconditions(cfg)
with launch_simulation(cfg, {"headless": True}):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    env.reset()
    pm = u._phase_machine
    ph0 = pm.phase.clone()
    zero = torch.zeros(N, 4, device=u.device)
    env.step(zero)
    ph1 = pm.phase.clone()
    tcp0, _ = tcp_and_rot(u)
    cube0, _ = cube_state(u)
    for _ in range(49):
        env.step(zero)
    tcp1, _ = tcp_and_rot(u)
    cube1, _ = cube_state(u)
    lf, rf, _, _ = contacts(u)
    held = (lf > 1.0) & (rf > 1.0)
    drift = torch.linalg.norm(tcp1 - tcp0, dim=-1)
    rel = torch.linalg.norm((cube1 - tcp1) - (cube0 - tcp0), dim=-1)
    rows = {}
    for k, name in enumerate(PHASES):
        m = ph0 == k
        if not m.any():
            continue
        row = {"n": int(m.sum()), "same_phase_after_1_step": round(float((ph1[m] == k).float().mean()), 3),
               "forward_after_1_step": round(float((ph1[m] > k).float().mean()), 3),
               "fingertip_drift_1s_m_median": round(float(drift[m].median()), 4)}
        if k >= LIFT:
            row["cube_held_both_pads_after_1s"] = round(float(held[m].float().mean()), 3)
            row["cube_vs_fingertip_slip_1s_m_max"] = round(float(rel[m].max()), 4)
            row["phase_after_1s_still_grasped"] = round(float(pm.grasp[m].float().mean()), 3)
        rows[name] = row
    res = {"task": TASK, "envs": N, "restored_fraction": 1.0, "per_restored_phase": rows,
           "nan_guard_events": u.nan_guard_events}
    (ROOT / "results" / "demo_reset_proof.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
