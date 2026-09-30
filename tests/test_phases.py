"""Unit tests for the section-16 phase machine (no simulator)."""

import math

import torch

from lift_rl.phases import CARRY, CLOSE, DESCEND, HOLD, HOVER, LIFT, PhaseMachine, PhaseParams

P = PhaseParams()
CUBE = torch.tensor([[0.5, 0.0, 0.0207]])
TARGET = torch.tensor([[0.5, 0.15, 0.35]])
PRE = torch.tensor([[0.5, 0.0, 0.0207 + 0.0207 + 0.10]])
GRASP = torch.tensor([[0.5, 0.0, 0.0207 + P.grasp_dz]])


def step(m, tcp, cube=CUBE, speed=0.0, tilt=0.0, yaw=0.0, lf=0.0, rf=0.0, palm=0.0, table=0.0):
    f = lambda v: torch.tensor([float(v)])  # noqa: E731
    return m.step(tcp, f(speed), f(tilt), f(yaw), cube, TARGET, f(lf), f(rf), f(palm), f(table))


def to_close(m):
    step(m, PRE)
    assert m.phase.item() == DESCEND
    for _ in range(P.sweet_steps):
        step(m, GRASP, speed=0.05)
    assert m.phase.item() == CLOSE


def test_forward_always_pays_more():
    for k in range(5):
        best_here = (k + 1) + P.progress_scale * 1.0
        worst_next = (k + 2) + P.progress_scale * 0.0
        assert worst_next > best_here


def test_full_forward_sequence_and_entry_bonus_once():
    m = PhaseMachine(1, "cpu")
    o = step(m, PRE)
    assert m.phase.item() == DESCEND and o["entry"].item() == 1.0
    for _ in range(P.sweet_steps):
        o = step(m, GRASP, speed=0.05)
    assert m.phase.item() == CLOSE and o["entry"].item() == 1.0
    o = step(m, GRASP, lf=6, rf=6)
    assert m.phase.item() == LIFT and m.grasp.item()
    lifted = CUBE + torch.tensor([[0, 0, 0.11]])
    step(m, GRASP + torch.tensor([[0, 0, 0.11]]), cube=lifted, lf=6, rf=6)
    assert m.phase.item() == CARRY
    near = TARGET + torch.tensor([[0, 0.01, 0]])
    o = step(m, near, cube=near, lf=6, rf=6)
    assert m.phase.item() == HOLD and o["entry"].item() == 1.0
    o = step(m, TARGET, cube=TARGET, lf=6, rf=6)
    assert o["entry"].item() == 0.0 and o["hold"].item() == 1.0
    assert m.entered.all()


def test_sweet_spot_needs_slow_vertical_aligned():
    for kw in ({"speed": 0.2}, {"tilt": math.radians(15)}, {"yaw": math.radians(15)}):
        m = PhaseMachine(1, "cpu")
        step(m, PRE)
        for _ in range(10):
            step(m, GRASP, **kw)
        assert m.phase.item() == DESCEND, kw
    m = PhaseMachine(1, "cpu")
    step(m, PRE)
    for _ in range(10):
        step(m, GRASP + torch.tensor([[0.02, 0, 0]]))  # 2 cm off in xy (tolerance 1.5 cm)
    assert m.phase.item() == DESCEND


def test_close_reopens_when_leaving_or_no_contact():
    m = PhaseMachine(1, "cpu")
    to_close(m)
    step(m, GRASP + torch.tensor([[0, 0, 0.03]]))  # hand leaves the band before both pads touch
    assert m.phase.item() == DESCEND
    m = PhaseMachine(1, "cpu")
    to_close(m)
    for _ in range(P.close_no_contact_steps):
        step(m, GRASP)
    assert m.phase.item() == DESCEND


def test_palm_blocks_confirmation():
    m = PhaseMachine(1, "cpu")
    to_close(m)
    step(m, GRASP, lf=6, rf=6, palm=3)
    assert m.phase.item() == CLOSE and not m.grasp.item()


def test_drop_goes_back_to_hover_and_bonus_not_repaid():
    m = PhaseMachine(1, "cpu")
    to_close(m)
    step(m, GRASP, lf=6, rf=6)
    assert m.phase.item() == LIFT
    o = None
    for _ in range(P.drop_steps + 1):
        o = step(m, GRASP, lf=0, rf=0)
    assert m.phase.item() == HOVER and not m.grasp.item() and o["drop"].item() == 1.0
    o = step(m, PRE)
    assert m.phase.item() == DESCEND and o["entry"].item() == 0.0


def test_push_penalty_only_before_first_grasp():
    m = PhaseMachine(1, "cpu")
    step(m, PRE + torch.tensor([[0.2, 0, 0]]))  # spawn position recorded on the first step
    o = step(m, PRE + torch.tensor([[0.2, 0, 0]]), cube=CUBE + torch.tensor([[0.02, 0, 0]]))
    assert o["push"].item() == 1.0


def test_penalty_indicators():
    m = PhaseMachine(1, "cpu")
    o = step(m, torch.tensor([[0.5, 0.0, 0.05]]), speed=0.5, table=25, palm=2)
    assert o["table"].item() == 1.0 and o["palm"].item() == 1.0
    assert abs(o["low_fast"].item() - 0.25) < 1e-6


def test_reset():
    m = PhaseMachine(2, "cpu")
    m.phase[:] = HOLD
    m.grasp[:] = True
    m.entered[:] = True
    m.reset(torch.tensor([1]))
    assert m.phase.tolist() == [HOLD, HOVER] and m.grasp.tolist() == [True, False]
    assert m.entered[1].tolist() == [True, False, False, False, False, False]


def test_descend_progress_has_no_speed_gate_and_is_coarse_plus_fine():
    m = PhaseMachine(1, "cpu")
    step(m, PRE)
    far = GRASP + torch.tensor([[0, 0, 0.05]])
    slow = step(m, far, speed=0.0)["progress"].item()
    fast = step(m, far, speed=0.5)["progress"].item()
    assert abs(slow - fast) < 1e-6 and slow > 0.0
    near = step(m, GRASP + torch.tensor([[0, 0, 0.005]]), speed=0.5)["progress"].item()
    assert near > slow
    d = 0.005
    expect = 0.5 * (1 - math.tanh(d / 0.05)) + 0.5 * (1 - math.tanh(d / 0.01))
    assert abs(near - expect) < 1e-5


def test_near_fast_penalty_smooth_and_only_near_the_cube():
    m = PhaseMachine(1, "cpu")
    low = GRASP + torch.tensor([[0, 0, 0.03]])  # fingertips 3 cm above the cube center -> < 5 cm above its top
    assert step(m, low, speed=0.05)["near_fast"].item() == 0.0
    assert abs(step(m, low, speed=0.30)["near_fast"].item() - 0.20) < 1e-6
    high = torch.tensor([[0.5, 0.0, 0.0207 + 0.0207 + 0.08]])
    assert step(m, high, speed=0.30)["near_fast"].item() == 0.0


def test_descend_progress_needs_vertical_and_aligned():
    m = PhaseMachine(1, "cpu")
    step(m, PRE)
    at = GRASP + torch.tensor([[0, 0, 0.02]])
    upright = step(m, at)["progress"].item()
    tilted = step(m, at, tilt=math.radians(20))["progress"].item()
    skewed = step(m, at, yaw=math.radians(20))["progress"].item()
    assert tilted < upright and skewed < upright
    assert step(m, at, tilt=math.radians(30))["progress"].item() == 0.0


def test_hover_progress_pays_for_approach_even_far_away():
    m = PhaseMachine(1, "cpu")
    far = step(m, PRE + torch.tensor([[-0.25, 0, 0.25]]), tilt=math.radians(40))["progress"].item()
    farther = step(m, PRE + torch.tensor([[-0.35, 0, 0.35]]), tilt=math.radians(40))["progress"].item()
    assert far - farther > 0.02  # the 0.5 m term keeps a gradient at 35-50 cm


def test_near_fast_off_while_grasped():
    m = PhaseMachine(1, "cpu")
    to_close(m)
    step(m, GRASP, lf=6, rf=6)
    assert m.grasp.item()
    assert step(m, GRASP + torch.tensor([[0, 0, 0.02]]), speed=0.3, lf=6, rf=6)["near_fast"].item() == 0.0
