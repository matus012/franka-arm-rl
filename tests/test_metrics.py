"""Success / lift / slam / jam detectors on crafted states (no simulator)."""

from __future__ import annotations

import torch

from lift_rl.metrics import CUBE_REST_Z, EpisodeTracker, Thresholds, summarize, wilson_ci

Z = torch.zeros


def _t(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_success_and_lift_at_episode_end():
    tr = EpisodeTracker(4, "cpu")
    tgt = _t(0.5, 0.0, 0.30).repeat(4, 1)
    cube = torch.stack([
        _t(0.5, 0.0, 0.29),  # 1 cm off, in the air      -> success, lift
        _t(0.5, 0.04, 0.30),  # 4 cm off                  -> 5 cm success only, lift
        _t(0.5, 0.0, CUBE_REST_Z),  # on the table        -> nothing
        _t(0.5, 0.0, 0.30),  # at target but table contact reported -> not success (lift still true)
    ])
    table_f = _t(0.0, 0.0, 5.0, 5.0)
    out = tr.final(cube, tgt, table_f)
    assert out["success"].tolist() == [True, False, False, False]
    assert out["success_5cm"].tolist() == [True, True, False, False]
    assert out["lift"].tolist() == [True, True, False, True]


def test_lift_threshold_is_rest_plus_5cm():
    tr = EpisodeTracker(2, "cpu")
    cube = torch.stack([_t(0, 0, CUBE_REST_Z + 0.049), _t(0, 0, CUBE_REST_Z + 0.051)])
    out = tr.final(cube, cube.clone(), Z(2))
    assert out["lift"].tolist() == [False, True]


def test_slam_any_step_above_threshold():
    th = Thresholds()
    tr = EpisodeTracker(3, "cpu", th)
    for f in ([0.0, 5.0, 25.0], [0.0, 19.9, 0.0]):
        tr.update(_t(*f), Z(3), Z(3), Z(3), Z(3))
    out = tr.final(Z(3, 3), Z(3, 3), Z(3))
    assert out["slam"].tolist() == [False, False, True]
    assert out["slam_10N"].tolist() == [False, True, True]
    assert out["slam_50N"].tolist() == [False, False, False]
    assert torch.allclose(out["max_table_force"], _t(0.0, 19.9, 25.0))


def test_jam_needs_consecutive_steps():
    th = Thresholds(jam_steps=10)
    tr = EpisodeTracker(3, "cpu", th)
    lifted = _t(*[CUBE_REST_Z + 0.10] * 3)
    for t in range(12):
        palm = _t(5.0 if t < 12 else 0, 5.0 if t % 2 == 0 else 0.0, 0.0)  # env0 sustained, env1 intermittent
        both = _t(10.0, 10.0, 10.0)
        tr.update(Z(3), lifted, palm, both, both)
    out = tr.final(Z(3, 3), Z(3, 3), Z(3))
    assert out["jam"].tolist() == [True, False, False]


def test_jam_one_finger_scoop_but_clean_pinch_ok():
    tr = EpisodeTracker(3, "cpu", Thresholds(jam_steps=3))
    lifted = _t(*[CUBE_REST_Z + 0.10] * 3)
    for _ in range(3):
        tr.update(
            Z(3),
            lifted,
            palm_force=Z(3),
            lf_force=_t(20.0, 20.0, 0.0),  # env0 pinch, env1 one finger, env2 nothing touching (e.g. thrown)
            rf_force=_t(20.0, 0.0, 0.0),
        )
    out = tr.final(Z(3, 3), Z(3, 3), Z(3))
    assert out["jam"].tolist() == [False, True, False]


def test_cube_on_table_pushed_by_one_finger_is_not_jam():
    tr = EpisodeTracker(1, "cpu", Thresholds(jam_steps=2))
    for _ in range(5):
        tr.update(Z(1), _t(CUBE_REST_Z), Z(1), _t(20.0), Z(1))
    assert not tr.final(Z(1, 3), Z(1, 3), Z(1))["jam"].item()


def test_nan_invalid_episode_counts_as_failure():
    tr = EpisodeTracker(1, "cpu")
    tr.update(Z(1), Z(1), Z(1), Z(1), Z(1), nan_flag=torch.tensor([True]))
    cube = _t(0.5, 0, 0.3)[None]
    out = tr.final(cube, cube.clone(), Z(1))
    assert not out["success"].item() and out["invalid"].item()


def test_wilson_and_summary():
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_ci(0, 1000)[0] == 0.0
    tr = EpisodeTracker(2, "cpu")
    out = tr.final(torch.stack([_t(0, 0, .3), _t(0, 0, .1)]), _t(0, 0, .3).repeat(2, 1), Z(2))
    s = summarize(out)
    assert s["episodes"] == 2 and s["success"]["count"] == 1
    assert abs(s["final_dist_median_m"] - 0.1) < 1e-6


def test_jam_cube_riding_on_wrist():
    """Audit finding: a lifted cube touching only an arm link (wrist) is a jam; on the table it is not."""
    tr = EpisodeTracker(2, "cpu", Thresholds(jam_steps=3))
    z = _t(CUBE_REST_Z + 0.15, CUBE_REST_Z)
    for _ in range(4):
        tr.update(Z(2), z, Z(2), Z(2), Z(2), arm_force=_t(15.0, 15.0))
    assert tr.final(Z(2, 3), Z(2, 3), Z(2))["jam"].tolist() == [True, False]
