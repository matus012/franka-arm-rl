"""Simulator tests (GPU). Each check runs the real script in a subprocess (one sim per process).

  uv run pytest tests/test_sim.py      (~6 min: builds small envs, a 2-iteration training run, eval + play smoke)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OWN_OBS_DIM = 9 + 9 + 3 + 6 + 7 + 8  # joint pos, joint vel, cube pos, cube orientation (quat + sin/cos 4 yaw), target pose command (pos + quat), last action


def run(args: list[str], timeout: int = 900) -> str:
    r = subprocess.run([PY, "-u", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def check_env(task: str) -> dict:
    out = run(["scripts/check_env.py", "--task", task, "--num_envs", "8", "--steps", "60"])
    line = [ln for ln in out.splitlines() if ln.startswith("CHECK_ENV=")][-1]
    return json.loads(line[len("CHECK_ENV="):])


@pytest.mark.parametrize("task,obs_dim", [("LiftRL-Own-v0", OWN_OBS_DIM), ("LiftRL-Twist-v0", OWN_OBS_DIM),
                                          ("LiftRL-Staged-v0", OWN_OBS_DIM), ("LiftRL-Staged-DR-v0", OWN_OBS_DIM),
                                          ("LiftRL-Staged-AutoClose-v0", OWN_OBS_DIM), ("LiftRL-Frozen-DR-v0", OWN_OBS_DIM),
                                          ("LiftRL-Stock-Newton-v0", 36)])
def test_env_shapes_dtypes_and_reward_ranges(task, obs_dim):
    r = check_env(task)
    assert r["obs_shape"] == [8, obs_dim] and r["step_obs_shape"] == [8, obs_dim]
    assert r["obs_dtype"] == "torch.float32" and r["reward_dtype"] == "torch.float32"
    assert r["action_dim"] == 8
    assert r["reward_shape"] == [8]
    assert r["terminated_dtype"] == "torch.bool" and r["truncated_dtype"] == "torch.bool"
    assert r["all_finite"], r
    assert r["out_of_range"] == [], r["reward_term_ranges"]
    assert r["unknown_terms"] == []


@pytest.fixture(scope="module")
def tiny_checkpoint() -> Path:
    out = run(["scripts/train.py", "--task", "LiftRL-Own-v0", "--num_envs", "32", "--max_iterations", "2",
               "--seed", "1", "--headless", "--experiment_name", "pytest", "--run_name", "tiny"])
    log_dir = Path([ln for ln in out.splitlines() if ln.startswith("[lift_rl] log_dir=")][-1].split("=", 1)[1])
    ckpts = sorted(log_dir.glob("model_*.pt"))
    assert ckpts, f"no checkpoint in {log_dir}"
    return ckpts[-1]


def test_eval_script_smoke(tiny_checkpoint, tmp_path):
    out_json = tmp_path / "eval.json"
    run(["scripts/evaluate.py", "--task", "LiftRL-Own-v0", "--checkpoint", str(tiny_checkpoint),
         "--out", str(out_json), "--num_envs", "4", "--num_episodes", "4", "--seed", "3"])
    r = json.loads(out_json.read_text())
    assert r["episodes"] == 4
    for k in ("success", "success_5cm", "lift", "slam", "jam"):
        assert 0.0 <= r[k]["rate"] <= 1.0 and len(r[k]["ci95"]) == 2
    assert r["final_dist_median_m"] >= 0.0


def test_play_script_smoke(tiny_checkpoint):
    out = run(["scripts/play.py", "--task", "LiftRL-Own-v0", "--num_envs", "4", "--checkpoint", str(tiny_checkpoint),
               "--max_steps", "30", "--headless"])
    assert "[lift_rl] played 30 steps, obs finite: True" in out


def test_nan_guard_recovers_one_world():
    out = run(["scripts/check_nan_guard.py", "--task", "LiftRL-Own-v0"])
    r = json.loads([ln for ln in out.splitlines() if ln.startswith("CHECK_NAN=")][-1][len("CHECK_NAN="):])
    assert r["events"] == 1 and r["flag"] == [False, False, False, True, False, False, False, False]
    assert r["reward_bad"] == 0.0 and r["terminated_bad"] and not r["terminated_others"] and r["reward_finite"]
    assert r["events_after_30_steps"] == 1 and r["finite_after"] and r["solver_finite_after"]


def test_phased_env_scripted_arm_with_reflex_reaches_hold(tmp_path):
    """Section 16: LiftRL-Phased-Fixed-v0 (51-dim obs, 7 arm actions, reflex gripper): the step-A controller reaches
    HOLD and succeeds in every episode; the rake never gets past HOVER."""
    out = tmp_path / "clean.json"
    run(["scripts/phased_proofs.py", "--mode", "clean", "--num_envs", "8", "--seed", "3", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["funnel_pct"]["hold"] == 100.0 and r["success"]["rate"] == 1.0 and r["table_contact"]["rate"] == 0.0
    out = tmp_path / "rake.json"
    run(["scripts/phased_proofs.py", "--mode", "rake", "--num_envs", "8", "--seed", "3", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["funnel_pct"]["descend"] == 0.0 and r["return_mean"] < 0.0
