"""Staged reward ordering (brief 10.2) and clean-grasp detector (brief 10.4) on crafted states."""

from __future__ import annotations

import math

import torch

from lift_rl.metrics import CUBE_REST_Z, EpisodeTracker, Thresholds
from lift_rl.staged import CUBE_HALF, StagedParams, low_fast_penalty, staged_reward, yaw_mod90_error

P = StagedParams()
CUBE = torch.tensor([[0.5, 0.0, CUBE_REST_Z]])
TGT = torch.tensor([[0.5, 0.1, 0.35]])


def _r(ee, speed=0.0, tilt_deg=0.0, yaw_deg=0.0, cube=CUBE, lf=0.0, rf=0.0, palm=0.0, closed=False, target=TGT):
    t = lambda v: torch.tensor([float(v)])  # noqa: E731
    return staged_reward(torch.tensor([ee], dtype=torch.float32), t(speed), t(math.radians(tilt_deg)),
                         t(math.radians(yaw_deg)), cube, target, t(lf), t(rf), t(palm), torch.tensor([closed]), P)


def test_grasp_pays_only_after_a_descent_over_the_cube():
    t = lambda v: torch.tensor([float(v)])  # noqa: E731
    lifted = torch.tensor([[0.5, 0.0, CUBE_REST_Z + 0.2]])
    ee = torch.tensor([[0.5, 0.0, CUBE_REST_Z + 0.2]])
    args = (ee, t(0.0), t(0.0), t(0.0), lifted, TGT, t(20), t(20), t(0), torch.tensor([True]), P)
    assert staged_reward(*args, descended=torch.tensor([False]))["stage"].item() <= 1  # raked / scooped: no descent
    assert staged_reward(*args, descended=torch.tensor([True]))["stage"].item() == 4
    # a slow aligned descent to grasp height sets the latch; a fast one does not
    low = torch.tensor([[0.5, 0.0, CUBE_REST_Z + 0.01]])
    base = (low, None, t(0.0), t(0.0), CUBE, TGT, t(0), t(0), t(0), torch.tensor([False]), P)
    for speed, want in ((0.1, True), (0.3, False)):
        a = list(base)
        a[1] = t(speed)
        assert staged_reward(*a, descended=torch.tensor([False]))["descended"].item() is want
    # latch released when the hand leaves the cube without holding it
    away = torch.tensor([[0.6, 0.0, 0.2]])
    assert not staged_reward(away, t(0.0), t(0.0), t(0.0), CUBE, TGT, t(0), t(0), t(0), torch.tensor([False]), P,
                             descended=torch.tensor([True]))["descended"].item()


def test_each_stage_reached_by_the_right_state():
    pre_z = CUBE_REST_Z + CUBE_HALF + 0.10
    assert _r([0.3, 0.2, 0.4])["stage"].item() == 0  # far away
    assert _r([0.5, 0.0, pre_z])["stage"].item() == 1  # over the cube, aligned
    assert _r([0.5, 0.0, pre_z], tilt_deg=25)["stage"].item() == 0  # over but tilted
    assert _r([0.5, 0.0, pre_z], yaw_deg=20)["stage"].item() == 0  # over but fingers not on faces
    assert _r([0.51, 0.0, pre_z])["stage"].item() == 1 and _r([0.52, 0.0, pre_z])["stage"].item() == 0  # xy 1.5 cm
    assert _r([0.5, 0.0, CUBE_REST_Z], lf=20, rf=20, closed=True)["stage"].item() == 2
    assert _r([0.5, 0.0, CUBE_REST_Z], lf=20, rf=20, closed=False)["stage"].item() == 1  # not commanded closed
    # probe 4: commanding close at grasp height over the cube is already stage 2 (before the pads touch)
    closing = _r([0.5, 0.0, CUBE_REST_Z], closed=True)
    assert closing["stage"].item() == 2 and closing["progress"].item() == 0.5
    assert _r([0.5, 0.0, CUBE_REST_Z + 0.05], closed=True)["stage"].item() == 1  # closing above the grasp band: no
    assert _r([0.5, 0.0, CUBE_REST_Z], lf=20, rf=20, palm=5, closed=True)["stage"].item() == 1  # palm touches
    lifted = torch.tensor([[0.5, 0.0, CUBE_REST_Z + 0.03]])
    assert _r([0.5, 0.0, CUBE_REST_Z + 0.03], cube=lifted, lf=20, rf=20, closed=True)["stage"].item() == 3
    high = torch.tensor([[0.5, 0.0, CUBE_REST_Z + 0.2]])
    assert _r([0.5, 0.0, CUBE_REST_Z + 0.2], cube=high, lf=20, rf=20, closed=True)["stage"].item() == 4
    # rake / launch / wedge: cube in the air without two pads -> never stage 3/4
    assert _r([0.5, 0.0, CUBE_REST_Z + 0.2], cube=high, lf=20, rf=0, closed=True)["stage"].item() <= 1
    assert _r([0.5, 0.0, CUBE_REST_Z + 0.2], cube=high, palm=30, closed=True)["stage"].item() <= 1


def test_every_stage_pays_more_than_any_earlier_stage():
    g = torch.Generator().manual_seed(0)
    n = 20000
    ee = torch.rand(n, 3, generator=g) * torch.tensor([0.6, 0.6, 0.6]) + torch.tensor([0.2, -0.3, 0.0])
    cube = torch.rand(n, 3, generator=g) * torch.tensor([0.2, 0.5, 0.4]) + torch.tensor([0.4, -0.25, CUBE_REST_Z])
    cube[: n // 2, 2] = CUBE_REST_Z
    ee[: n // 4, :2] = cube[: n // 4, :2] + 0.01 * (torch.rand(n // 4, 2, generator=g) - 0.5)
    out = staged_reward(ee, torch.rand(n, generator=g) * 0.4, torch.rand(n, generator=g) * 0.6,
                        torch.rand(n, generator=g) * 0.78, cube, torch.rand(n, 3, generator=g),
                        torch.rand(n, generator=g) * 4, torch.rand(n, generator=g) * 4,
                        torch.rand(n, generator=g) * 2, torch.rand(n, generator=g) > 0.3, P)
    st, r = out["stage"], out["reward"]
    seen = sorted(set(st.tolist()))
    assert seen == [0, 1, 2, 3, 4], seen
    for k in range(1, 5):
        assert r[st == k].min() > r[st < k].max(), k
    assert torch.isfinite(r).all() and (out["progress"] >= 0).all() and (out["progress"] <= 1).all()


def test_descent_progress_paid_only_when_slow():
    z = CUBE_REST_Z + 0.05
    slow, fast = _r([0.5, 0.0, z], speed=0.10), _r([0.5, 0.0, z], speed=0.30)
    closed = _r([0.5, 0.0, z], speed=0.10, closed=True)
    assert slow["stage"].item() == fast["stage"].item() == closed["stage"].item() == 1
    assert slow["progress"].item() > 0.5 and fast["progress"].item() == 0.0 and closed["progress"].item() == 0.0
    lower = _r([0.5, 0.0, CUBE_REST_Z + 0.01], speed=0.1)
    assert lower["reward"].item() > slow["reward"].item()
    # at grasp height, commanding "close" jumps to stage 2 (probe 4): a full stage above staying open
    bottom_open = _r([0.5, 0.0, CUBE_REST_Z], speed=0.05)
    bottom_closing = _r([0.5, 0.0, CUBE_REST_Z], speed=0.05, closed=True)
    assert bottom_open["stage"].item() == 1 and bottom_closing["stage"].item() == 2
    assert bottom_closing["reward"].item() > bottom_open["reward"].item() + 0.5


def test_hold_bonus_only_near_target_in_stage_4():
    at_t = torch.tensor([[0.5, 0.1, 0.35]])
    r = _r([0.5, 0.1, 0.35], cube=at_t, lf=20, rf=20, closed=True)
    assert r["stage"].item() == 4 and r["bonus"].item()
    off = torch.tensor([[0.5, 0.1, 0.30]])
    assert not _r([0.5, 0.1, 0.30], cube=off, lf=20, rf=20, closed=True)["bonus"].item()


def test_yaw_error_mod_90_and_low_fast_penalty():
    y = lambda deg: torch.tensor([[math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0]])  # noqa: E731
    for hand, cube, want in ((0, 0, 0), (90, 0, 0), (100, 0, 10), (-40, 0, 40), (50, 0, 40), (30, 120, 0)):
        e = math.degrees(yaw_mod90_error(y(hand), torch.tensor([math.radians(cube)])).item())
        assert abs(e - want) < 1e-3, (hand, cube, e)
    z_low, z_high = torch.tensor([0.05]), torch.tensor([0.2])
    assert low_fast_penalty(z_low, torch.tensor([0.5])).item() > 0.24
    assert low_fast_penalty(z_low, torch.tensor([0.2])).item() == 0.0
    assert low_fast_penalty(z_high, torch.tensor([1.0])).item() == 0.0


# ---- clean grasp detector ----------------------------------------------------------------------
def _episode(steps):
    """steps: list of dicts with robot_table, ee_z, tilt, cube_z, lf, rf, palm (1 env)."""
    tr = EpisodeTracker(1, "cpu", Thresholds())
    t = lambda v: torch.tensor([float(v)])  # noqa: E731
    for s in steps:
        tr.update(t(s.get("hand_table", 0)), t(s["cube_z"]), t(s.get("palm", 0)), t(s.get("lf", 0)), t(s.get("rf", 0)),
                  robot_table_force=t(s.get("robot_table", 0)), ee_z=t(s["ee_z"]), hand_tilt_deg=t(s.get("tilt", 0)),
                  cube_xy=torch.tensor([[0.5, s.get("cube_y", 0.0)]]))
    return tr.final(torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1))


CLEAN = [
    {"ee_z": 0.30, "cube_z": 0.055},  # spawn: the cube drops from 3.4 cm above rest
    {"ee_z": 0.25, "cube_z": 0.04},
    {"ee_z": 0.20, "cube_z": CUBE_REST_Z},
    {"ee_z": 0.06, "cube_z": CUBE_REST_Z, "tilt": 5},
    {"ee_z": 0.03, "cube_z": CUBE_REST_Z, "tilt": 8},
    {"ee_z": 0.021, "cube_z": CUBE_REST_Z, "tilt": 8, "lf": 30, "rf": 30},
    {"ee_z": 0.05, "cube_z": CUBE_REST_Z + 0.03, "tilt": 40, "lf": 30, "rf": 30},  # tilt after lift-off is fine
]


def test_scripted_style_clean_grasp_is_clean():
    out = _episode(CLEAN)
    assert out["clean"].item() and out["liftoff"].item()


def test_each_clean_criterion_can_fail_alone():
    table = [dict(s) for s in CLEAN]
    table[4]["robot_table"] = 25  # arm/hand hits the table
    tilted = [dict(s) for s in CLEAN]
    tilted[4]["tilt"] = 35  # lunging, tilted hand low
    one_pad = [dict(s) for s in CLEAN]
    one_pad[6]["rf"] = 0  # lift-off with one pad (scoop / rake)
    palm = [dict(s) for s in CLEAN]
    palm[5]["palm"] = 3
    no_lift = CLEAN[:6]
    for name, ep, crit in (("table", table, "clean_c1_no_table_hit"), ("tilt", tilted, "clean_c2_top_down"),
                           ("one_pad", one_pad, "clean_c3_both_pads_at_liftoff"), ("palm", palm, "clean_c4_no_palm"),
                           ("no_lift", no_lift, "clean_c3_both_pads_at_liftoff")):
        out = _episode(ep)
        assert not out["clean"].item(), name
        assert not out[crit].item(), name


def test_high_tilt_outside_descent_zone_is_ignored():
    ep = [dict(s) for s in CLEAN]
    ep[2]["tilt"] = 60  # far above the cube: not the last 5 cm of descent
    assert _episode(ep)["clean"].item()


def test_spawn_drop_is_not_liftoff():
    """Probe-1 bug: the cube falling after spawn was counted as lift-off, which also switched off c2."""
    ep = [{"ee_z": 0.30, "cube_z": 0.055}, {"ee_z": 0.03, "cube_z": CUBE_REST_Z, "tilt": 45}]
    out = _episode(ep)
    assert not out["liftoff"].item() and not out["clean_c2_top_down"].item()
