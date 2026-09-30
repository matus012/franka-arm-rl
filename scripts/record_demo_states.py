"""17.3d: record the step-A scripted controller's states (through the 4-D task-space interface, reflex gripper) in the
fixed world, every 5th control step of 64 episodes (seed 21, not an eval seed) -> data/demo_states_fixed_ts.pt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.demo_reset import snapshot  # noqa: E402
from lift_rl.evaluation import make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import Thresholds, summarize  # noqa: E402
from lift_rl.phases import PHASES  # noqa: E402
from lift_rl.physics import assert_run_preconditions  # noqa: E402
from lift_rl.scripted import ScriptedController  # noqa: E402
from lift_rl.task_space import scripted_delta_policy  # noqa: E402

import argparse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="LiftRL-Phased-Fixed-TS-v0")
ap.add_argument("--out", default=str(ROOT / "data" / "demo_states_fixed_ts.pt"))
ap.add_argument("--num_envs", type=int, default=64)
args = ap.parse_args()
TASK = args.task
cfg = make_eval_cfg(load_cfg_from_registry(TASK, "env_cfg_entry_point"), args.num_envs, 21)
assert_run_preconditions(cfg)
with launch_simulation(cfg, {"headless": True}):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    ctrl = ScriptedController(env, Thresholds(), reflex=True)
    ctrl.started = False
    pol = scripted_delta_policy(ctrl, u.action_manager.get_term("arm_action"))
    snaps = []

    def on_step(t, uu):
        if t % 5 == 4:
            snaps.append(snapshot(uu))

    out = rollout(env, pol, Thresholds(), on_step=on_step)
    ok = out["success"].cpu()
    bank = {k: torch.cat([s[k][ok] for s in snaps]) for k in snaps[0]}
    counts = [int((bank["phase"] == k).sum()) for k in range(6)]
    (ROOT / "data").mkdir(exist_ok=True)
    torch.save(bank, args.out)
    print(f"episodes {len(ok)}, successful {int(ok.sum())}, snapshots {int(bank['phase'].numel())}, per phase "
          + ", ".join(f"{PHASES[k]} {counts[k]}" for k in range(6)))
    print("summary", {k: summarize(out)[k]["rate"] for k in ("success", "clean")})
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
