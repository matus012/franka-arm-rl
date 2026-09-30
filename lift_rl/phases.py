"""Section 16.1: the one phase machine (HOVER -> DESCEND -> CLOSE -> LIFT -> CARRY -> HOLD) shared by the reward, the
observations, the gripper reflex, the evaluator and the videos. Pure torch on per-env tensors (unit-testable
without a simulator); lift_rl/phased_cfg.py feeds it the sim state once per control step.

Forward transitions (one per step):
  HOVER   -> DESCEND  fingertip center within 1.5 cm of the pre-grasp point (10 cm above the cube top), tilt < 10 deg,
                      yaw error to the cube faces < 10 deg
  DESCEND -> CLOSE    sweet spot held for 3 steps: tilt < 10, yaw < 10, fingertip center within 1 cm of the cube center
                      in xy, fingertip height = cube center + GRASP_DZ +- 1 cm, fingertip speed < 0.10 m/s
  CLOSE   -> LIFT     both pads > 5 N and no palm contact -> grasp confirmed (latched)
  LIFT    -> CARRY    cube >= 10 cm above its rest height
  CARRY   -> HOLD     cube within 2 cm of the target
Backward:
  CLOSE -> DESCEND (reopen): the hand leaves the sweet-spot band (speed not required) before both pads touch, or no
                      pad contact within 0.5 s, or no confirmed grasp within 1 s (palm stuck on the cube; safety
                      rule added during development)
  any grasped phase -> HOVER (drop): fewer than two pads touching (> 1 N) for > 5 steps; the gripper opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

HOVER, DESCEND, CLOSE, LIFT, CARRY, HOLD = range(6)
PHASES = ("hover", "descend", "close", "lift", "carry", "hold")
CUBE_HALF = 0.0207  # resting cube-center height = half the cube edge [m] (metrics.CUBE_REST_Z)


@dataclass
class PhaseParams:
    pregrasp_dz: float = 0.10  # pre-grasp point: fingertip center 10 cm above the cube top
    hover_tol: float = 0.015
    tilt_ok: float = math.radians(10.0)
    yaw_ok: float = math.radians(10.0)
    sweet_xy: float = 0.015  # run 2: 1.0 -> 1.5 cm (noise-tested with the scripted arm)
    # fingertip-center height above the cube center at the grasp, MEASURED from the step-A scripted grasp
    # (scripts/phased_proofs.py --mode clean, 200 episodes: +0.2 mm median at grasp confirmation, q05 -1.0 mm; palm-to-
    # cube-top gap 1.7 cm; results/phased_proof_clean.json). The band is +-1 cm around it.
    grasp_dz: float = 0.0
    sweet_dz: float = 0.015  # run 2: +-1.0 -> +-1.5 cm
    sweet_speed: float = 0.15  # run 2: entry speed 0.10 -> 0.15 m/s
    sweet_steps: int = 2  # run 2: 3 -> 2 steps
    pad_confirm_n: float = 5.0
    contact_n: float = 1.0
    close_no_contact_steps: int = 25  # 0.5 s
    close_max_steps: int = 50  # 1 s
    lift_dz: float = 0.10
    carry_tol: float = 0.02
    drop_steps: int = 5
    # reward (stage units per step, before Isaac Lab's dt)
    progress_scale: float = 0.9
    entry_bonus: float = 2.0
    hold_bonus_r: float = 0.01
    push_tol: float = 0.015
    table_n: float = 20.0
    low_z: float = 0.10
    fast_v: float = 0.25
    near_v: float = 0.10  # near_fast penalty threshold (17.3c), kept at 0.10 when the sweet-spot entry speed moved
    near_dz: float = 0.05  # near_fast penalty zone: fingertips < 5 cm above the cube top


class PhaseMachine:
    def __init__(self, num_envs: int, device, p: PhaseParams | None = None):
        self.n, self.dev, self.p = num_envs, device, p or PhaseParams()
        z = lambda dt=torch.long: torch.zeros(num_envs, dtype=dt, device=device)  # noqa: E731
        self.phase = z()
        self.grasp = z(torch.bool)
        self.ever_grasped = z(torch.bool)
        self.sweet_run, self.close_t, self.lost_run = z(), z(), z()
        self.entered = torch.zeros(num_envs, 6, dtype=torch.bool, device=device)
        self.entered[:, HOVER] = True
        self.spawn_xy = torch.zeros(num_envs, 2, device=device)
        self.spawn_set = z(torch.bool)
        self.drops = z()
        self.lf = torch.zeros(num_envs, device=device)
        self.rf = torch.zeros(num_envs, device=device)
        self.out: dict[str, torch.Tensor] = {}

    def reset(self, ids=None) -> None:
        ids = slice(None) if ids is None else ids
        self.phase[ids] = HOVER
        for t in (self.grasp, self.ever_grasped, self.spawn_set):
            t[ids] = False
        for t in (self.sweet_run, self.close_t, self.lost_run, self.drops):
            t[ids] = 0
        self.entered[ids] = False
        self.entered[ids, HOVER] = True
        self.lf[ids] = 0.0
        self.rf[ids] = 0.0

    def step(self, tcp, speed, tilt, yaw_err, cube, target, lf, rf, palm_f, table_f) -> dict[str, torch.Tensor]:
        """All (N,) or (N,3) tensors in the env frame; forces in N, angles in rad, speed in m/s. Returns the reward
        components (unweighted indicators / values) and the phase quantities."""
        p = self.p
        self.spawn_xy = torch.where(self.spawn_set[:, None], self.spawn_xy, cube[:, :2])
        self.spawn_set |= True
        self.lf, self.rf = lf, rf
        pre = torch.cat([cube[:, :2], (cube[:, 2] + CUBE_HALF + p.pregrasp_dz)[:, None]], 1)
        d_pre = torch.linalg.norm(tcp - pre, dim=-1)
        aligned = (tilt < p.tilt_ok) & (yaw_err < p.yaw_ok)
        grasp_pt = torch.cat([cube[:, :2], (cube[:, 2] + p.grasp_dz)[:, None]], 1)
        xy_err = torch.linalg.norm(tcp[:, :2] - cube[:, :2], dim=-1)
        dz = tcp[:, 2] - grasp_pt[:, 2]
        in_band = aligned & (xy_err < p.sweet_xy) & (dz.abs() < p.sweet_dz)
        sweet = in_band & (speed < p.sweet_speed)
        both1 = (lf > p.contact_n) & (rf > p.contact_n)
        any1 = (lf > p.contact_n) | (rf > p.contact_n)
        both5 = (lf > p.pad_confirm_n) & (rf > p.pad_confirm_n)
        palm = palm_f > p.contact_n
        cube_h = cube[:, 2] - CUBE_HALF
        dist = torch.linalg.norm(cube - target, dim=-1)

        old = self.phase.clone()
        self.sweet_run = torch.where((old == DESCEND) & sweet, self.sweet_run + 1, torch.zeros_like(self.sweet_run))
        self.close_t = torch.where(old == CLOSE, self.close_t + 1, torch.zeros_like(self.close_t))
        to_desc = (old == HOVER) & (d_pre < p.hover_tol) & aligned
        to_close = (old == DESCEND) & (self.sweet_run >= p.sweet_steps)
        confirm = (old == CLOSE) & both5 & ~palm
        reopen = (old == CLOSE) & ~confirm & (
            (~in_band & ~both1) | ((self.close_t >= p.close_no_contact_steps) & ~any1) | (self.close_t >= p.close_max_steps)
        )
        to_carry = (old == LIFT) & (cube_h >= p.lift_dz)
        to_hold = (old == CARRY) & (dist < p.carry_tol)
        self.lost_run = torch.where(self.grasp & ~both1, self.lost_run + 1, torch.zeros_like(self.lost_run))
        drop = self.grasp & (self.lost_run > p.drop_steps)

        ph = old
        for m, k in ((to_desc, DESCEND), (to_close, CLOSE), (confirm, LIFT), (reopen, DESCEND), (to_carry, CARRY),
                     (to_hold, HOLD), (drop, HOVER)):
            ph = torch.where(m, torch.full_like(ph, k), ph)
        self.phase = ph
        self.grasp = (self.grasp | confirm) & ~drop
        self.ever_grasped |= confirm
        self.lost_run = torch.where(drop, torch.zeros_like(self.lost_run), self.lost_run)
        self.drops += drop.long()
        first = ~self.entered.gather(1, ph[:, None])[:, 0] & (ph != old)
        self.entered[torch.arange(self.n, device=self.dev), ph] = True

        # progress within the phase, in [0, 1]
        prog = torch.zeros(self.n, device=self.dev)
        tilt_f = (1 - tilt / math.radians(30.0)).clamp(0, 1)
        yaw_f = (1 - yaw_err / math.radians(45.0)).clamp(0, 1)
        # 17.3d probe A: + a long-range term (0.5 m) so moving toward the cube always pays (run 3 retreated where the
        # 0.1 m term was flat)
        prog_h = 0.3 * (1 - torch.tanh(d_pre / 0.5)) + 0.3 * (1 - torch.tanh(d_pre / 0.10)) + 0.2 * tilt_f + 0.2 * yaw_f
        d_sweet = torch.linalg.norm(tcp - grasp_pt, dim=-1)
        # 17.3c: no speed gate on DESCEND progress (the gate made freezing optimal); coarse (5 cm) + fine (1 cm).
        # Run 2 (2026-09-29): times an alignment factor; run 1b's policy parked tilted 30 deg / yaw 12 deg 7 cm above
        # the grasp point, because nothing in DESCEND paid for staying vertical and aligned
        align = (1 - tilt / math.radians(30.0)).clamp(0, 1) * (1 - yaw_err / math.radians(30.0)).clamp(0, 1)
        prog_d = (0.5 * (1 - torch.tanh(d_sweet / 0.05)) + 0.5 * (1 - torch.tanh(d_sweet / 0.01))) * align
        prog_c = 0.5 * (lf / p.pad_confirm_n).clamp(0, 1) + 0.5 * (rf / p.pad_confirm_n).clamp(0, 1)
        prog_l = (cube_h / p.lift_dz).clamp(0, 1)
        prog_ca = 0.5 * (1 - torch.tanh(dist / 0.3)) + 0.5 * (1 - torch.tanh(dist / 0.05))
        prog_ho = 1 - torch.tanh(dist / 0.02)
        for k, v in ((HOVER, prog_h), (DESCEND, prog_d), (CLOSE, prog_c), (LIFT, prog_l), (CARRY, prog_ca), (HOLD, prog_ho)):
            prog = torch.where(ph == k, v, prog)
        pushed = ~self.ever_grasped & (torch.linalg.norm(cube[:, :2] - self.spawn_xy, dim=-1) > p.push_tol)
        self.out = {
            "phase": ph,
            "progress": prog,
            "phase_level": (ph + 1).float() + p.progress_scale * prog,
            "entry": (first & (ph >= DESCEND)).float(),
            "hold": ((ph == HOLD) & (dist < p.hold_bonus_r)).float(),
            "table": (table_f > p.table_n).float(),
            "palm": palm.float(),
            "push": pushed.float(),
            "drop": drop.float(),
            "low_fast": torch.relu(speed - p.fast_v) * (tcp[:, 2] < p.low_z).float(),
            # 17.3c: smooth slow-down near the cube instead of the progress gate
            # run 5: only before a confirmed grasp (it is an approach rule; run 4 was penalized for any carry faster
            # than 0.10 m/s because the zone moves with the held cube)
            "near_fast": torch.relu(speed - p.near_v) * ((tcp[:, 2] - (cube[:, 2] + CUBE_HALF) < p.near_dz) & ~self.grasp).float(),
            "sweet": sweet,
            "in_band": in_band,
            "dist": dist,
            "grasp": self.grasp.clone(),
        }
        return self.out


def phase_one_hot(phase: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(phase, 6).float()
