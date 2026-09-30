"""Section 16.7 step 0 proofs on LiftRL-Phased-Fixed-v0 (fixed world, phase machine, gripper reflex).

  --mode clean  the step-A controller, arm only, the reflex closes the claw. Must reach HOLD in >= 95 % of 200
                episodes. Also MEASURES the grasp pose (fingertip-center height above the cube center and the
                palm-to-cube-top gap when the phase machine enters CLOSE) and the squeeze slip (cube-vs-fingertip
                offset change after the grasp is confirmed, through lift, carry and hold at step-A speeds).
  --mode rake   the run-1 rake (tilted lunge + drag). Must never get past DESCEND and must be penalized.
Writes results/phased_proof_<mode>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=("clean", "rake"), required=True)
p.add_argument("--num_envs", type=int, default=200)
p.add_argument("--seed", type=int, default=12345)
p.add_argument("--debug_env", type=int, default=-1)
p.add_argument("--out", default=None)
p.add_argument("--task", default="LiftRL-Phased-Fixed-v0", help="LiftRL-Phased-Fixed-TS-v0: scripted arm through the 4-D task-space actions")
p.add_argument("--action_noise", type=float, default=0.0,
               help="Gaussian noise std added to the 7 arm actions each step (policy-like exploration noise)")
args = p.parse_args()

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import lift_rl  # noqa: E402,F401
from isaaclab_tasks.utils import launch_simulation  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from lift_rl.evaluation import make_eval_cfg, rollout  # noqa: E402
from lift_rl.metrics import CUBE_REST_Z, Thresholds, summarize  # noqa: E402
from lift_rl.phased_cfg import tcp_and_rot  # noqa: E402
from lift_rl.phases import CLOSE, LIFT, PHASES  # noqa: E402
from lift_rl.physics import assert_run_preconditions  # noqa: E402
from lift_rl.scripted import RakeController, ScriptedController  # noqa: E402

TASK = args.task
HULL_BOTTOM_ABOVE_TCP = 0.1034 - 0.066  # panda_hand convex hull: flat bottom 6.6 cm below the hand origin

cfg = make_eval_cfg(load_cfg_from_registry(TASK, "env_cfg_entry_point"), args.num_envs, args.seed)
assert_run_preconditions(cfg)
th = Thresholds()

with launch_simulation(cfg, {"headless": True}):
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    assert str(u.device).startswith("cuda"), u.device
    N = u.num_envs
    ctrl = (ScriptedController if args.mode == "clean" else RakeController)(env, th, reflex=True, debug_env=args.debug_env)
    ret = torch.zeros(N, device=u.device)
    terms = {k: torch.zeros(N, device=u.device) for k in u.reward_manager.active_terms}
    rec = {"close_dz": torch.full((N,), float("nan"), device=u.device), "prev_phase": None,
           "confirm_dz": torch.full((N,), float("nan"), device=u.device),
           "offset0": torch.full((N, 3), float("nan"), device=u.device), "slip": torch.zeros(N, device=u.device),
           "max_phase": torch.zeros(N, dtype=torch.long, device=u.device)}

    def on_step(t, uu):
        global ret
        ret += uu.reward_buf
        for k in terms:
            terms[k] += uu.reward_manager._step_reward[:, uu.reward_manager.active_terms.index(k)] * uu.step_dt
        pm = uu._phase_machine
        tcp, _ = tcp_and_rot(uu)
        cube = uu.scene["object"].data.root_pos_w
        cube = (cube.torch if hasattr(cube, "torch") else cube) - uu.scene.env_origins
        prev = rec["prev_phase"] if rec["prev_phase"] is not None else torch.zeros_like(pm.phase)
        new_close = (pm.phase == CLOSE) & (prev != CLOSE) & torch.isnan(rec["close_dz"])
        rec["close_dz"] = torch.where(new_close, tcp[:, 2] - cube[:, 2], rec["close_dz"])
        new_grasp = (pm.phase == LIFT) & (prev == CLOSE) & torch.isnan(rec["offset0"][:, 0])
        off = cube - tcp
        rec["offset0"] = torch.where(new_grasp[:, None], off, rec["offset0"])
        rec["confirm_dz"] = torch.where(new_grasp, tcp[:, 2] - cube[:, 2], rec["confirm_dz"])
        slip = torch.linalg.norm(off - rec["offset0"], dim=-1)
        rec["slip"] = torch.where(pm.grasp & ~torch.isnan(slip), torch.maximum(rec["slip"], slip), rec["slip"])
        rec["prev_phase"] = pm.phase.clone()
        rec["max_phase"] = torch.maximum(rec["max_phase"], pm.phase)

    ctrl.started = False
    base = ctrl
    if "-TS-" in TASK:
        from lift_rl.task_space import scripted_delta_policy

        base = scripted_delta_policy(ctrl, u.action_manager.get_term("arm_action"))
    policy = base if args.action_noise <= 0 else (lambda o: (lambda a: a + args.action_noise * torch.randn_like(a))(base(o)))
    out = rollout(env, policy, th, on_step=on_step)
    s = summarize(out)
    q = lambda x, qs=(0.05, 0.5, 0.95): [round(float(torch.quantile(x, v)), 4) for v in qs]  # noqa: E731
    cd = rec["close_dz"][~torch.isnan(rec["close_dz"])]
    gs = rec["slip"][~torch.isnan(rec["offset0"][:, 0])]
    res = {"mode": args.mode, "action_noise": args.action_noise, "task": TASK, "episodes": N, "seed": args.seed, "episode_steps": cfg.eval_steps,
           "funnel_pct": {n: round(100 * s[f"reached_{n}"]["rate"], 1) for n in PHASES},
           "max_phase_counts": [int((rec["max_phase"] == k).sum()) for k in range(6)],
           "final_phase_counts": s["final_phase_counts"], "episodes_with_drop": s["episodes_with_drop"],
           **{k: s[k] for k in ("success", "success_5cm", "clean", "lift", "slam", "jam", "table_contact")},
           "final_dist_median_m": s["final_dist_median_m"],
           "return_mean": float(ret.mean()), "return_quantiles_5_50_95": q(ret),
           "return_by_term_mean": {k: round(float(v.mean()), 3) for k, v in terms.items()},
           "push_before_liftoff_median_m": s.get("push_before_liftoff_median_m"),
           "nan_guard_events": u.nan_guard_events}
    if len(cd):
        res["grasp_pose_measured"] = {
            "tcp_minus_cube_center_z_m_q05_50_95": q(cd),
            "at_grasp_confirmation_tcp_minus_cube_center_z_m_q05_50_95": q(rec["confirm_dz"][~torch.isnan(rec["confirm_dz"])]),
            "at_grasp_confirmation_palm_gap_m_q05_50_95": q(rec["confirm_dz"][~torch.isnan(rec["confirm_dz"])] + HULL_BOTTOM_ABOVE_TCP - CUBE_REST_Z),
            "palm_gap_m_q05_50_95": q(cd + HULL_BOTTOM_ABOVE_TCP - CUBE_REST_Z),
            "note": "palm gap = hand-hull bottom (TCP + 3.74 cm) - cube top (cube center + 2.07 cm), at CLOSE entry"}
    if len(gs):
        res["squeeze_slip_m_q05_50_95_max"] = q(gs) + [round(float(gs.max()), 4)]
    dst = Path(args.out) if args.out else ROOT / "results" / f"phased_proof_{args.mode}.json"
    dst.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))
    print(f"NAN_GUARD_EVENTS={u.nan_guard_events}")
    env.close()
