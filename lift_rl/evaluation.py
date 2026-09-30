"""Evaluation env config + rollout loop shared by scripts/evaluate.py, scripts/make_videos.py and tests.

Eval conditions (brief section 6): full 250-step episodes (no early termination, no timeout inside
the rollout), no mid-episode command resampling, no observation noise, NaN-guard events counted,
deterministic (mean) actions. "Episode end" = the state after the 250th control step.
"""

from __future__ import annotations

import torch
from isaaclab.utils.math import combine_frame_transforms

from .instrument import HAND_BODIES, add_eval_sensors
from .metrics import EpisodeTracker, Thresholds

EPISODE_STEPS = 250  # 5 s at 50 Hz


def make_eval_cfg(env_cfg, num_envs: int, seed: int, sensors: bool = True):
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env_cfg.sim.device = "cuda:0"
    env_cfg.episode_length_s = 1000.0  # no timeout inside the rollout; we stop at EPISODE_STEPS
    env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.object_pose.debug_vis = False
    env_cfg.observations.policy.enable_corruption = False
    for name in list(vars(env_cfg.terminations)):
        if name != "time_out" and not name.startswith("_"):
            setattr(env_cfg.terminations, name, None)
    if getattr(env_cfg.events, "demo_reset", None) is not None:  # 17.3d: evaluation always starts from the normal pose
        env_cfg.events.demo_reset = None
    if env_cfg.curriculum is not None:  # curriculum only changes reward weights; irrelevant at eval
        env_cfg.curriculum = None
    if sensors:
        add_eval_sensors(env_cfg)
    return env_cfg


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def contact_readings(u) -> dict[str, torch.Tensor]:
    """Per-env contact-force magnitudes [N] from the eval sensors."""
    n = u.num_envs
    ht_sensor = u.scene["contact_hand_table"]
    rt_all = _t(ht_sensor.data.force_matrix_w).reshape(n, -1, 3).norm(dim=-1)  # (N, robot bodies)
    hand_rows = [i for i, name in enumerate(ht_sensor.sensor_names) if name in HAND_BODIES]
    ht = rt_all[:, hand_rows].amax(dim=1)
    ch_sensor = u.scene["contact_cube_hand"]
    ch = _t(ch_sensor.data.force_matrix_w).reshape(n, -1, 3).norm(dim=-1)
    idx = {name: i for i, name in enumerate(ch_sensor.filter_object_names)}
    ct = _t(u.scene["contact_cube_table"].data.force_matrix_w).reshape(n, -1, 3).norm(dim=-1).amax(dim=1)
    links = [i for name, i in idx.items() if name.startswith("panda_link")]
    return {
        "hand_table": ht,
        "robot_table": rt_all.amax(dim=1),
        "palm": ch[:, idx["panda_hand"]],
        "lf": ch[:, idx["panda_leftfinger"]],
        "rf": ch[:, idx["panda_rightfinger"]],
        "arm": ch[:, links].amax(dim=1) if links else torch.zeros_like(ht),
        "cube_table": ct,
    }


def cube_and_target(u) -> tuple[torch.Tensor, torch.Tensor]:
    """Cube center and commanded target, both in the env frame (N,3)."""
    robot, obj = u.scene["robot"], u.scene["object"]
    origins = u.scene.env_origins
    cmd = u.command_manager.get_command("object_pose")
    tgt_w, _ = combine_frame_transforms(_t(robot.data.root_pos_w), _t(robot.data.root_quat_w), cmd[:, :3])
    return _t(obj.data.root_pos_w) - origins, tgt_w - origins


def hand_pose(u) -> tuple[torch.Tensor, torch.Tensor]:
    """Fingertip-center z (env frame) and hand tilt from straight down [deg], both (N,)."""
    from isaaclab.utils.math import matrix_from_quat

    ee = _t(u.scene["ee_frame"].data.target_pos_w)[..., 0, :] - u.scene.env_origins
    robot = u.scene["robot"]
    q = _t(robot.data.body_quat_w)[:, robot.body_names.index("panda_hand")]
    z_axis = matrix_from_quat(q)[..., :, 2]
    tilt = torch.rad2deg(torch.arccos((-z_axis[:, 2]).clamp(-1.0, 1.0)))
    return ee[:, 2], tilt


@torch.inference_mode()
def rollout(env, policy, th: Thresholds | None = None, on_step=None) -> dict:
    """One batch of N full episodes from a fresh reset. Returns per-episode outcome tensors."""
    u = env.unwrapped
    obs = env.reset()[0]  # works for the gym env (obs, info) and the rsl_rl wrapper (obs, extras)
    tracker = EpisodeTracker(u.num_envs, u.device, th)
    steps = getattr(u.cfg, "eval_steps", EPISODE_STEPS)  # section 16 envs: 400 steps (8 s)
    for t in range(steps):
        actions = policy(obs)
        obs = env.step(actions)[0]
        c = contact_readings(u)
        cube, _ = cube_and_target(u)
        ee_z, tilt = hand_pose(u)
        tracker.update(c["hand_table"], cube[:, 2], c["palm"], c["lf"], c["rf"], getattr(u, "_nan_flag", None),
                       arm_force=c["arm"], robot_table_force=c["robot_table"], ee_z=ee_z, hand_tilt_deg=tilt,
                       cube_xy=cube[:, :2])
        if on_step is not None:
            on_step(t, u)
    cube, target = cube_and_target(u)
    out = tracker.final(cube, target, contact_readings(u)["cube_table"])
    pm = getattr(u, "_phase_machine", None)
    if pm is not None:  # section 16 phase funnel: phases reached during the episode, phase at the end, drops
        from .phases import PHASES

        for k, name in enumerate(PHASES):
            out[f"reached_{name}"] = pm.entered[:, k].clone()
        out["final_phase"] = pm.phase.clone()
        out["drops"] = pm.drops.clone()
    out["table_contact"] = out["max_robot_table_force"] > (th or Thresholds()).clean_table_n
    return out
