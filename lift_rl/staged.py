"""Staged pick reward (brief 10.2). Stages are judged from the state at every step, never from time.

    stage 0  hover    fingertip center -> pre-grasp point 10 cm above the cube top (coarse + fine kernel),
                      hand pointing down, fingers lined up with a pair of cube faces (yaw mod 90 deg)
                      (probe 4: the probe-2/3 open-gripper term is removed)
    stage 1  descent  hand over the cube (xy < 1.5 cm) and aligned (tilt < 20 deg, yaw < 15 deg): progress
                      of lowering toward grasp height, paid ONLY while hand speed <= 0.15 m/s; with the
                      gripper open (capped 0.95), or closing once at grasp height (1.0)
    stage 2  grasp    (probe 4) entered as soon as "close" is commanded at grasp height (h >= 0.9) over the cube
                      after the descent latch; progress 0.5 while closing, 1.0 once both pads are on the cube
                      (palm not touching). Stages 3-4 need the pads. The latch = the hand came down over the cube: a latch set by a slow (<= 0.15 m/s), aligned descent to
                      within 2 cm of grasp height over the cube (xy < 1.5 cm); cleared when the hand is not
                      grasping and more than 3 cm (xy) away from the cube. Stages 3-4 inherit it.
                      (Added 2026-09-29: without it, scripted rakes that end in a two-pad grasp reached
                      stage 4; development log, not published.)
    stage 3  lift     grasped and cube > 1 cm above rest: progress = cube height (full at +5 cm)
    stage 4  carry    grasped and cube > 5 cm above rest: cube-to-target tracking (coarse + fine)
                      + a separate per-step bonus inside 3 cm

Per-step stage reward = S + 0.9 * progress, progress in [0, 1]. So any state in stage S pays at least S,
and any earlier stage pays at most (S - 1) + 0.9 < S: moving forward always pays (unit-tested).
Always-on shaping (separate terms): action rate, joint velocity, fast-and-low hand penalty, table-hit
termination + penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .metrics import CUBE_REST_Z

CUBE_HALF = 0.024
TABLE_TOP_Z = CUBE_REST_Z - CUBE_HALF  # [m] env frame


@dataclass(frozen=True)
class StagedParams:
    pregrasp_above_top: float = 0.10
    over_xy: float = 0.015
    tilt_ok_deg: float = 20.0  # probe 2 (was 15)
    yaw_ok_deg: float = 15.0  # probe 2 (was 10)
    descent_speed_max: float = 0.15
    min_above_table: float = 0.01
    liftoff_dz: float = 0.01
    carry_dz: float = 0.05
    contact_n: float = 1.0
    progress_scale: float = 0.9
    hover_std: float = 0.10
    hover_std_fine: float = 0.02  # probe 2: near-field gradient (probe 1 froze a few cm off the cube)
    track_std_coarse: float = 0.30
    track_std_fine: float = 0.05
    bonus_radius: float = 0.03
    low_zone: float = 0.10  # "lower than 10 cm above the table"
    low_speed_max: float = 0.25
    latch_z_margin: float = 0.02
    latch_release_xy: float = 0.03


def yaw_mod90_error(hand_y_axis: torch.Tensor, cube_yaw: torch.Tensor) -> torch.Tensor:
    """|angle| between the finger closing axis (hand y-axis, projected to xy) and the nearest cube face
    normal, in [0, pi/4]."""
    psi_h = torch.atan2(hand_y_axis[:, 1], hand_y_axis[:, 0])
    d = psi_h - cube_yaw
    d = torch.remainder(d + math.pi / 4, math.pi / 2) - math.pi / 4
    return d.abs()


def staged_reward(
    ee: torch.Tensor,  # (N,3) fingertip center, env frame
    ee_speed: torch.Tensor,  # (N,) [m/s]
    tilt: torch.Tensor,  # (N,) hand z-axis vs straight down [rad]
    yaw_err: torch.Tensor,  # (N,) [rad], from yaw_mod90_error
    cube: torch.Tensor,  # (N,3) cube center, env frame
    target: torch.Tensor,  # (N,3) commanded target, env frame
    lf: torch.Tensor,  # (N,) left pad-cube force [N]
    rf: torch.Tensor,
    palm: torch.Tensor,
    grip_closed: torch.Tensor,  # (N,) bool, gripper commanded closed
    p: StagedParams = StagedParams(),
    descended: torch.Tensor | None = None,  # (N,) bool latch from previous steps (None = no latch required)
) -> dict[str, torch.Tensor]:
    """Returns stage (long), progress, reward (= stage + 0.9 progress), bonus (bool), grasped (bool) and the
    updated descent latch `descended`."""
    pre = cube.clone()
    pre[:, 2] = cube[:, 2] + CUBE_HALF + p.pregrasp_above_top
    d_pre = torch.linalg.norm(ee - pre, dim=-1)
    down = torch.cos(tilt).clamp(0.0, 1.0)
    yaw_align = 0.5 * (1.0 + torch.cos(4.0 * yaw_err))
    # probe 4: no open-gripper term in stage 0 (it pushed the gripper output to +1.3 and closing was never tried)
    phi0 = (0.35 * (1.0 - torch.tanh(d_pre / p.hover_std)) + 0.25 * (1.0 - torch.tanh(d_pre / p.hover_std_fine))
            + 0.20 * down + 0.20 * yaw_align)

    xy_err = torch.linalg.norm(ee[:, :2] - cube[:, :2], dim=-1)
    over = (xy_err < p.over_xy) & (tilt < math.radians(p.tilt_ok_deg)) & (yaw_err < math.radians(p.yaw_ok_deg))
    z_grasp = torch.clamp(cube[:, 2], min=TABLE_TOP_Z + p.min_above_table)
    h = ((pre[:, 2] - ee[:, 2]) / (pre[:, 2] - z_grasp).clamp(min=1e-3)).clamp(0.0, 1.0)
    # probe 3: open while descending; at grasp height (h >= 0.9) closing keeps (and slightly raises) the progress,
    # so there is no reward valley between "open at the bottom" and "pads on the cube" (probe 2 hovered open)
    at_bottom = h >= 0.9
    phi1 = torch.where(~grip_closed, h.clamp(max=0.95), torch.where(at_bottom, torch.ones_like(h), torch.zeros_like(h)))
    phi1 = torch.where(ee_speed <= p.descent_speed_max, phi1, torch.zeros_like(h))

    pads = (lf > p.contact_n) & (rf > p.contact_n) & (palm <= p.contact_n) & grip_closed
    on_table = cube[:, 2] - CUBE_REST_Z < p.liftoff_dz  # a descent is onto a cube still resting on the table
    descend_now = over & on_table & (ee_speed <= p.descent_speed_max) & (ee[:, 2] <= z_grasp + p.latch_z_margin)
    if descended is None:
        latch = torch.ones_like(pads)
    else:
        latch = (descended | descend_now) & ~(~pads & (xy_err > p.latch_release_xy))
    grasped = pads & latch
    # probe 4: stage 2 is entered as soon as "close" is commanded at grasp height over the cube after the descent
    # latch (immediate reward for closing); pad contact then raises the progress inside stage 2
    closing = latch & grip_closed & at_bottom & over & on_table & (palm <= p.contact_n)
    in_stage2 = closing | grasped
    dz = cube[:, 2] - CUBE_REST_Z
    phi2 = 0.5 + 0.5 * grasped.float()
    phi3 = (dz / p.carry_dz).clamp(0.0, 1.0)
    d_t = torch.linalg.norm(cube - target, dim=-1)
    phi4 = 0.5 * (1.0 - torch.tanh(d_t / p.track_std_coarse)) + 0.5 * (1.0 - torch.tanh(d_t / p.track_std_fine))

    stage = torch.zeros_like(h, dtype=torch.long)
    stage = torch.where(over, torch.ones_like(stage), stage)
    stage = torch.where(in_stage2, torch.full_like(stage, 2), stage)
    stage = torch.where(grasped & (dz > p.liftoff_dz), torch.full_like(stage, 3), stage)
    stage = torch.where(grasped & (dz > p.carry_dz), torch.full_like(stage, 4), stage)
    phis = torch.stack([phi0, phi1, phi2, phi3, phi4], dim=1)
    progress = phis.gather(1, stage[:, None])[:, 0]
    reward = stage.float() + p.progress_scale * progress
    bonus = (stage == 4) & (d_t < p.bonus_radius)
    return {"stage": stage, "progress": progress, "reward": reward, "bonus": bonus, "grasped": grasped,
            "d_target": d_t, "descended": latch, "at_bottom": at_bottom, "over": over, "on_table": on_table}


def low_fast_penalty(ee_z: torch.Tensor, ee_speed: torch.Tensor, p: StagedParams = StagedParams()) -> torch.Tensor:
    """Excess hand speed above 0.25 m/s while the fingertips are lower than 10 cm above the table [m/s]."""
    low = ee_z < TABLE_TOP_Z + p.low_zone
    return torch.where(low, (ee_speed - p.low_speed_max).clamp(min=0.0), torch.zeros_like(ee_speed))
