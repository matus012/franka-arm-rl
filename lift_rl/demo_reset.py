"""Section 17.3d: demonstration-state resets. The step-A scripted controller's states in the fixed world (recorded by
scripts/record_demo_states.py) are used ONLY as episode starting points: at reset a fraction of envs (default 50 %)
starts from a random snapshot, the phase of the snapshot drawn uniformly over the phases present; the rest start
normally. The policy is learned only by PPO. Evaluation always starts from the normal start pose
(evaluation.make_eval_cfg removes this event).

A snapshot holds: robot joint positions/velocities (9 + 9), cube pose (env frame) and velocity, and the phase-machine
state (phase, grasp latches, counters, entered flags, spawn xy, last pad forces). The gripper reflex re-latches its
squeeze from the restored pad forces on the first step; the task-space action term re-initializes its target from the
restored fingertip pose.
"""

from __future__ import annotations

from pathlib import Path

import torch

KEYS_ROBOT = ("joint_pos", "joint_vel")
KEYS_CUBE = ("cube_pos", "cube_quat", "cube_vel")
KEYS_PM = ("phase", "grasp", "ever_grasped", "sweet_run", "close_t", "lost_run", "entered", "spawn_xy", "drops", "lf", "rf")


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def snapshot(env) -> dict[str, torch.Tensor]:
    """Current state of every env (CPU tensors)."""
    u = env.unwrapped if hasattr(env, "unwrapped") else env
    r, obj, pm = u.scene["robot"], u.scene["object"], u._phase_machine
    s = {"joint_pos": _t(r.data.joint_pos), "joint_vel": _t(r.data.joint_vel),
         "cube_pos": _t(obj.data.root_pos_w) - u.scene.env_origins, "cube_quat": _t(obj.data.root_quat_w),
         "cube_vel": _t(obj.data.root_com_vel_w)}
    for k in KEYS_PM:
        s[k] = getattr(pm, k)
    return {k: v.detach().clone().cpu() for k, v in s.items()}


class DemoBank:
    def __init__(self, path: str | Path, device):
        d = torch.load(path, map_location=device)
        self.s = {k: v.to(device) for k, v in d.items()}
        self.n = int(self.s["phase"].shape[0])
        self.by_phase = [torch.nonzero(self.s["phase"] == k, as_tuple=False)[:, 0] for k in range(6)]
        self.present = [k for k in range(6) if len(self.by_phase[k])]

    def sample(self, m: int, device) -> torch.Tensor:
        ph = torch.tensor(self.present, device=device)[torch.randint(len(self.present), (m,), device=device)]
        idx = torch.empty(m, dtype=torch.long, device=device)
        for k in self.present:
            sel = ph == k
            if sel.any():
                pool = self.by_phase[k]
                idx[sel] = pool[torch.randint(len(pool), (int(sel.sum()),), device=device)]
        return idx


def restore_demo_states(env, env_ids, path: str, fraction: float = 0.5):
    """Reset event (must run after the normal reset events and reset_phase)."""
    if env_ids is None or len(env_ids) == 0 or fraction <= 0:
        return
    if not hasattr(env, "_demo_bank"):
        env._demo_bank = DemoBank(path, env.device)
        env._demo_restored = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    bank = env._demo_bank
    ids = torch.as_tensor(env_ids, device=env.device)
    env._demo_restored[ids] = False
    ids = ids[torch.rand(len(ids), device=env.device) < fraction]
    if len(ids) == 0:
        return
    k = bank.sample(len(ids), env.device)
    s = {key: v[k] for key, v in bank.s.items()}
    r, obj = env.scene["robot"], env.scene["object"]
    r.write_joint_state_to_sim_index(position=s["joint_pos"], velocity=s["joint_vel"], env_ids=ids)
    r.set_joint_position_target_index(target=s["joint_pos"], env_ids=ids)
    pose = torch.cat([s["cube_pos"] + env.scene.env_origins[ids], s["cube_quat"]], 1)
    obj.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
    obj.write_root_velocity_to_sim_index(root_velocity=s["cube_vel"], env_ids=ids)
    pm = env._phase_machine
    for key in KEYS_PM:
        getattr(pm, key)[ids] = s[key]
    pm.spawn_set[ids] = True
    env._demo_restored[ids] = True
