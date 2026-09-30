"""Section 12 checkpoint pick: score every saved checkpoint of a run on the validation seeds and pick one.

Rule (brief section 12): 200 validation episodes per checkpoint (eval seed 1001), rank by clean-grasp rate, ties
broken by success at 3 cm. One env is built once; before each checkpoint the RNG is reseeded, so every checkpoint
sees the same initial cube poses and targets (GPU physics itself is not bit-reproducible, gotcha 4a.5).

Usage: uv run python scripts/score_checkpoints.py --task LiftRL-Frozen-v0 --run_dir logs/rsl_rl/lift_rl/<run> \
           --out results/<name>_checkpoint_scores.json
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

import lift_rl  # noqa: F401
from isaaclab.envs import ManagerBasedEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import launch_simulation
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from lift_rl.evaluation import make_eval_cfg, rollout
from lift_rl.metrics import summarize
from lift_rl.physics import assert_on_cuda, assert_run_preconditions
from rsl_rl.runners import OnPolicyRunner

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--run_dir", default="")
p.add_argument("--ckpts", nargs="*", default=None, help="explicit checkpoint files instead of every model_*.pt in --run_dir")
p.add_argument("--out", required=True)
p.add_argument("--num_episodes", type=int, default=200)
p.add_argument("--seed", type=int, default=1001)
p.add_argument("--min_iter", type=int, default=0)
p.add_argument("--rank", choices=("clean", "success"), default="clean",
               help="clean: section 12 (max clean, ties success); success: section 16.5 (max success, ties clean)")
args = p.parse_args()

if args.ckpts:
    ckpts = [Path(c) for c in args.ckpts]
else:
    ckpts = sorted(Path(args.run_dir).glob("model_*.pt"), key=lambda f: int(re.findall(r"\d+", f.stem)[0]))
    ckpts = [c for c in ckpts if int(re.findall(r"\d+", c.stem)[0]) >= args.min_iter]
cfg = make_eval_cfg(load_cfg_from_registry(args.task, "env_cfg_entry_point"), args.num_episodes, args.seed)
assert_run_preconditions(cfg)

with launch_simulation(cfg, {"headless": True}):
    env = gym.make(args.task, cfg=cfg)
    agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"),
                                             metadata.version("rsl-rl-lib"))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    rows = []
    t0 = time.time()
    for ck in ckpts:
        runner.load(str(ck), map_location=agent_cfg.device)
        module = runner.get_inference_policy(device=env.unwrapped.device)
        assert_on_cuda(env, module)
        ManagerBasedEnv.seed(args.seed)  # same validation episodes for every checkpoint
        s = summarize(rollout(wrapped, lambda o: module(o)))
        it = int(re.findall(r"\d+", ck.stem)[0])
        rows.append({"iteration": it, "checkpoint": str(ck.resolve().relative_to(ROOT)),
                     **{k: s[k]["rate"] for k in ("clean", "success", "success_5cm", "lift", "slam", "jam", "liftoff")},
                     **({"table_contact": s["table_contact"]["rate"]} if "table_contact" in s else {}),
                     **({"funnel": {k[8:]: s[k]["rate"] for k in s if k.startswith("reached_")}} if "reached_hover" in s else {}),
                     "clean_ci95": s["clean"]["ci95"], "final_dist_median_m": s["final_dist_median_m"]})
        print(f"[score] it {it:5d} clean {rows[-1]['clean']:.3f} success {rows[-1]['success']:.3f} "
              f"lift {rows[-1]['lift']:.3f} slam {rows[-1]['slam']:.3f} jam {rows[-1]['jam']:.3f}", flush=True)
    best = max(rows, key=(lambda r: (r["clean"], r["success"])) if args.rank == "clean" else (lambda r: (r["success"], r["clean"])))
    rec = {"recorded_at": datetime.now().isoformat(timespec="seconds"), "task": args.task, "run_dir": args.run_dir,
           "rule": ("section 12: max validation clean rate, ties -> success at 3 cm" if args.rank == "clean"
                    else "section 16.5: max validation success at 3 cm, ties -> clean rate"), "val_seed": args.seed,
           "episodes_per_checkpoint": args.num_episodes, "pick": best, "scores": rows,
           "nan_guard_events": env.unwrapped.nan_guard_events, "wall_s": round(time.time() - t0, 1)}
    Path(args.out).write_text(json.dumps(rec, indent=2))
    print("PICK=" + json.dumps(best))
    print(f"NAN_GUARD_EVENTS={env.unwrapped.nan_guard_events}")
    env.close()
