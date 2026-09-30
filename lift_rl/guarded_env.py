"""ManagerBasedRLEnv with a NaN guard for Newton / MuJoCo-Warp (gotcha 4a.4).

Rare solver/contact blow-ups put NaN into one world's solver data. Resetting that env is not
enough because MuJoCo-Warp keeps the NaN in its per-world buffers (warm start, accelerations,
constraint arrays). The guard runs right after physics, before terminations and rewards:

1. detect worlds with non-finite qpos / qvel / qacc or non-finite robot / object state,
2. zero the non-finite entries of every float array in the solver data and the Newton state,
3. re-run forward kinematics and resync MuJoCo qpos/qvel from the (now finite) Newton state,
4. force a terminal step with reward 0 for the affected envs; they are then reset normally.

``step`` is ManagerBasedRLEnv.step from Isaac Lab 3.0.0b2 (BSD-3-Clause, Copyright (c) 2022-2026,
The Isaac Lab Project Developers) with the guard inserted after the physics loop.
"""

from __future__ import annotations

import dataclasses

import torch
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvStepReturn


def _float_warp_arrays(obj, depth: int = 0):
    """Yield every float-typed warp array reachable from a mujoco_warp Data dataclass."""
    if depth > 2 or obj is None:
        return
    items = (
        ((f.name, getattr(obj, f.name, None)) for f in dataclasses.fields(obj))
        if dataclasses.is_dataclass(obj)
        else vars(obj).items()
    )
    for _, v in items:
        if isinstance(v, wp.array):
            scalar = getattr(v.dtype, "_wp_scalar_type_", v.dtype)  # vec/mat/transform -> component type
            if v.size > 0 and scalar in (wp.float32, wp.float64, wp.float16):
                yield v
        elif dataclasses.is_dataclass(v):
            yield from _float_warp_arrays(v, depth + 1)


class GuardedRLEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv + NaN guard. Guard events are counted in ``nan_guard_events``."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.nan_guard_events: int = 0
        self._nan_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # -- guard -----------------------------------------------------------------------------------
    def _solver(self):
        from isaaclab_newton.physics import NewtonManager

        return NewtonManager._solver, NewtonManager

    def _detect_nonfinite(self) -> torch.Tensor:
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        robot = self.scene["robot"]
        obj = self.scene["object"]
        for t in (robot.data.joint_pos, robot.data.joint_vel, obj.data.root_pos_w, obj.data.root_lin_vel_w):
            t = t.torch if hasattr(t, "torch") else t
            bad |= ~torch.isfinite(t.reshape(self.num_envs, -1)).all(dim=1)
        solver, _ = self._solver()
        data = getattr(solver, "mjw_data", None)
        if data is not None:
            for name in ("qpos", "qvel", "qacc"):
                arr = wp.to_torch(getattr(data, name))
                if arr.shape[0] == self.num_envs:
                    bad |= ~torch.isfinite(arr.reshape(self.num_envs, -1)).all(dim=1)
        return bad

    def _sanitize(self) -> None:
        from newton import eval_fk

        solver, mgr = self._solver()
        arrays = list(_float_warp_arrays(getattr(solver, "mjw_data", None)))
        state = mgr._state_0
        for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_f"):
            a = getattr(state, name, None)
            if isinstance(a, wp.array) and a.size > 0:
                arrays.append(a)
        for a in arrays:
            t = wp.to_torch(a)
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0) if t.is_floating_point() else None
        eval_fk(mgr._model, state.joint_q, state.joint_qd, state, None)
        if getattr(solver, "mjw_data", None) is not None:
            solver._update_mjc_data(solver.mjw_data, mgr._model, state)
        # invalidate cached sensor / asset buffers so rewards read the sanitized state
        self.scene.update(dt=0.0)

    def _scrub_views(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Cached torch views (shared memory) of the arrays the next physics step reads.
        rows: per-world (nworld, k) arrays used for detection; scrub: arrays zeroed of non-finite entries."""
        if getattr(self, "_views", None) is None:
            solver, mgr = self._solver()
            data = getattr(solver, "mjw_data", None)
            rows, scrub = [], []
            if data is not None:
                for name in ("qpos", "qvel", "qacc", "qacc_warmstart"):
                    t = wp.to_torch(getattr(data, name))
                    scrub.append(t)
                    if name != "qacc_warmstart" and t.shape[0] == self.num_envs:
                        rows.append(t.reshape(self.num_envs, -1))
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                a = getattr(mgr._state_0, name, None)
                if isinstance(a, wp.array) and a.size > 0:
                    scrub.append(wp.to_torch(a))
            self._views = (rows, scrub)
        return self._views

    def _scrub_physics_step(self) -> None:
        """After every physics step, with NO host sync: flag worlds with non-finite solver state and zero
        the non-finite entries the next collision pass / solver step would read. A NaN born in the first
        physics step of a control step must not reach Newton's collision kernels in the second one (that
        crashed fix-2 training with a CUDA fault in run 1)."""
        rows, scrub = self._scrub_views()
        for t in rows:
            self._nan_flag |= ~torch.isfinite(t).all(dim=1)
        for t in scrub:
            torch.nan_to_num_(t, nan=0.0, posinf=0.0, neginf=0.0)

    def _nan_guard(self) -> None:
        """Once per control step (one host sync): count flagged worlds and run the full sanitize (every
        solver array, FK, qpos/qvel resync, cache refresh) if any world was flagged."""
        self._nan_flag |= self._detect_nonfinite()
        if bool(self._nan_flag.any()):
            self.nan_guard_events += int(self._nan_flag.sum().item())
            self._sanitize()

    # -- step (Isaac Lab ManagerBasedRLEnv.step + guard) -----------------------------------------
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.is_rendering
        self._nan_flag.zero_()

        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self._scrub_physics_step()
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self._scrub_physics_step()
                self.recorder_manager.record_post_physics_decimation_step()
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                self.scene.update(dt=self.physics_dt)

        self._nan_guard()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if self._nan_flag.any():
            self.reset_terminated = self.reset_terminated | self._nan_flag
            self.reset_time_outs = self.reset_time_outs & ~self._nan_flag
            self.reset_buf = self.reset_buf | self._nan_flag
            self.reward_buf = torch.where(self._nan_flag, torch.zeros_like(self.reward_buf), self.reward_buf)
        self.extras.setdefault("log", {})["Metrics/nan_guard_events_total"] = float(self.nan_guard_events)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)

        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
