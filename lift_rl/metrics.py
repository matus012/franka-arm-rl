"""Evaluation detectors (brief section 6). Pure torch on per-env tensors, so they can be unit-tested
on crafted states without a simulator.

Definitions (pre-registered 2026-09-28, before any policy was evaluated):
- success@r   : at episode end, |cube_center - target| < r and the cube does not touch the table
                (cube-table contact force <= CONTACT_N). r = 3 cm headline, 5 cm also reported.
- lift        : at episode end, cube center z >= resting center z + 5 cm.
- slam        : at any control step, the largest hand/finger-vs-table contact force > SLAM_N.
                SLAM_N = 20 N, about 3x the weight of hand + fingers (0.75 kg): resting a fingertip on
                the table stays below it, driving the hand into the table exceeds it. Sensitivity at
                10 N and 50 N is reported alongside.
- jam         : the cube is wedged against the hand instead of held between the fingertips, for at
                least JAM_STEPS consecutive control steps (0.2 s; filters transient brushes). A step is
                a jam step if
                  (a) cube-vs-palm (panda_hand body) contact force > CONTACT_N, or
                  (b) the cube is off the table (center z > rest + 2 cm), touches the robot (hand,
                      fingers or any arm link), and is NOT touched by both fingers (one-finger scoop,
                      cube riding on the wrist / forearm).
                (Arm links added 2026-09-29 after the by-eye audit found cubes carried on the wrist that
                the hand+finger-only sensor could not see; see the development log, not published.)
- final distance: |cube_center - target| at episode end, median over episodes.

Run 2 (brief 10.4, 2026-09-29) — clean grasp, per episode, ALL of:
  (c1) no hand/finger/ARM-vs-table contact force > 20 N at any step (all robot links, not only the hand);
  (c2) during the last 5 cm of descent the hand z-axis is within 30 deg of straight down. "Last 5 cm of
       descent" = every step before lift-off in which the fingertip center is lower than the resting cube
       center + 5 cm (whatever its xy position, so a low sideways lunge counts);
  (c3) both finger pads touch the cube (> 1 N each) at lift-off. Lift-off = first step with the cube center
       > 1 cm above rest, after the cube has come to rest once (it spawns 3.4 cm up and drops; counting that
       drop as lift-off was a bug found in probe 1, fixed 2026-09-29). No lift-off = not clean;
  (c4) the palm (panda_hand body) never touches the cube (> 1 N).
Diagnostic only (not part of "clean"): cube xy displacement before lift-off (pushed / raked cubes move).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

CUBE_REST_Z = 0.0207  # resting cube-center height above the env origin [m], measured (results/m0_scripted_grasp_stock.json, rest_z_m)


@dataclass
class Thresholds:
    success_r: float = 0.03
    success_r_loose: float = 0.05
    lift_dz: float = 0.05
    off_table_dz: float = 0.02
    slam_n: float = 20.0
    slam_n_sensitivity: tuple[float, ...] = (10.0, 50.0)
    contact_n: float = 1.0
    jam_steps: int = 10
    clean_table_n: float = 20.0
    clean_tilt_deg: float = 30.0
    clean_descent_zone: float = 0.05
    liftoff_dz: float = 0.01

    def to_dict(self) -> dict:
        return asdict(self)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial rate k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


class EpisodeTracker:
    """Accumulates per-step detector state for N parallel episodes."""

    def __init__(self, num_envs: int, device: str | torch.device, th: Thresholds | None = None):
        self.th = th or Thresholds()
        z = lambda dt=torch.bool: torch.zeros(num_envs, dtype=dt, device=device)  # noqa: E731
        self.slam = z()
        self.slam_sens = {f: z() for f in self.th.slam_n_sensitivity}
        self.max_table_force = z(torch.float32)
        self.jam = z()
        self.jam_run = z(torch.int32)
        self.max_jam_run = z(torch.int32)
        self.invalid = z()  # NaN-guard reset mid-episode
        # clean grasp (10.4)
        self.max_robot_table_force = z(torch.float32)
        self.tilt_violation = z()
        self.max_descent_tilt = z(torch.float32)
        self.liftoff = z()
        self.liftoff_both_pads = z()
        self.palm_ever = z()
        self.cube_xy0: torch.Tensor | None = None
        self.settled = z()  # the cube spawns 3.4 cm up and drops; lift-off only counts after it has rested
        self.push_before_liftoff = z(torch.float32)

    def jam_step(
        self,
        cube_z: torch.Tensor,
        palm_force: torch.Tensor,
        lf_force: torch.Tensor,
        rf_force: torch.Tensor,
        arm_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        th = self.th
        palm = palm_force > th.contact_n
        lf, rf = lf_force > th.contact_n, rf_force > th.contact_n
        arm = arm_force > th.contact_n if arm_force is not None else torch.zeros_like(palm)
        off_table = cube_z > CUBE_REST_Z + th.off_table_dz
        touching = palm | lf | rf | arm
        return palm | (off_table & touching & ~(lf & rf))

    def update(
        self,
        hand_table_force: torch.Tensor,
        cube_z: torch.Tensor,
        palm_force: torch.Tensor,
        lf_force: torch.Tensor,
        rf_force: torch.Tensor,
        nan_flag: torch.Tensor | None = None,
        arm_force: torch.Tensor | None = None,
        robot_table_force: torch.Tensor | None = None,
        ee_z: torch.Tensor | None = None,
        hand_tilt_deg: torch.Tensor | None = None,
        cube_xy: torch.Tensor | None = None,
    ) -> None:
        """All inputs (N,): contact-force magnitudes [N] and cube-center z relative to env origin [m].
        arm_force = largest cube-vs-arm-link (panda_link0..7) contact force.
        Clean-grasp inputs (optional): robot_table_force = largest contact force of ANY robot body vs the
        table; ee_z = fingertip-center z (env frame); hand_tilt_deg = angle between the hand z-axis and
        straight down; cube_xy (N,2) = cube center xy (env frame)."""
        self.max_table_force = torch.maximum(self.max_table_force, hand_table_force)
        self.slam |= hand_table_force > self.th.slam_n
        for f in self.slam_sens:
            self.slam_sens[f] |= hand_table_force > f
        j = self.jam_step(cube_z, palm_force, lf_force, rf_force, arm_force)
        self.jam_run = torch.where(j, self.jam_run + 1, torch.zeros_like(self.jam_run))
        self.max_jam_run = torch.maximum(self.max_jam_run, self.jam_run)
        self.jam |= self.jam_run >= self.th.jam_steps
        if nan_flag is not None:
            self.invalid |= nan_flag
        # -- clean grasp
        th = self.th
        rt = robot_table_force if robot_table_force is not None else hand_table_force
        self.max_robot_table_force = torch.maximum(self.max_robot_table_force, rt)
        self.palm_ever |= palm_force > th.contact_n
        newly_settled = ~self.settled & (cube_z < CUBE_REST_Z + 0.005)
        if cube_xy is not None:
            if self.cube_xy0 is None:
                self.cube_xy0 = cube_xy.clone()
            self.cube_xy0 = torch.where(newly_settled[:, None], cube_xy, self.cube_xy0)  # xy where it came to rest
            moved = torch.linalg.norm(cube_xy - self.cube_xy0, dim=-1)
            self.push_before_liftoff = torch.where(
                self.liftoff | ~self.settled, self.push_before_liftoff, torch.maximum(self.push_before_liftoff, moved)
            )
        self.settled |= newly_settled
        new_lift = ~self.liftoff & self.settled & (cube_z > CUBE_REST_Z + th.liftoff_dz)
        both = (lf_force > th.contact_n) & (rf_force > th.contact_n)
        self.liftoff_both_pads |= new_lift & both
        self.liftoff |= new_lift
        if ee_z is not None and hand_tilt_deg is not None:
            in_zone = ~self.liftoff & (ee_z < CUBE_REST_Z + th.clean_descent_zone)  # before lift-off
            self.tilt_violation |= in_zone & (hand_tilt_deg > th.clean_tilt_deg)
            self.max_descent_tilt = torch.where(
                in_zone, torch.maximum(self.max_descent_tilt, hand_tilt_deg), self.max_descent_tilt
            )

    def clean(self) -> torch.Tensor:
        th = self.th
        return (
            (self.max_robot_table_force <= th.clean_table_n)
            & ~self.tilt_violation
            & self.liftoff
            & self.liftoff_both_pads
            & ~self.palm_ever
            & ~self.invalid
        )

    def final(self, cube_pos: torch.Tensor, target_pos: torch.Tensor, cube_table_force: torch.Tensor) -> dict:
        """Per-episode outcome tensors at episode end. cube_pos/target_pos (N,3) in the env frame."""
        th = self.th
        dist = torch.linalg.norm(cube_pos - target_pos, dim=-1)
        on_table = cube_table_force > th.contact_n
        ok = ~self.invalid
        return {
            "dist": dist,
            "success": (dist < th.success_r) & ~on_table & ok,
            "success_5cm": (dist < th.success_r_loose) & ~on_table & ok,
            "lift": (cube_pos[:, 2] >= CUBE_REST_Z + th.lift_dz) & ok,
            "slam": self.slam,
            "jam": self.jam,
            "invalid": self.invalid,
            "max_table_force": self.max_table_force,
            "max_jam_run": self.max_jam_run,
            **{f"slam_{int(f)}N": v for f, v in self.slam_sens.items()},
            "clean": self.clean(),
            "clean_c1_no_table_hit": self.max_robot_table_force <= th.clean_table_n,
            "clean_c2_top_down": ~self.tilt_violation,
            "clean_c3_both_pads_at_liftoff": self.liftoff & self.liftoff_both_pads,
            "clean_c4_no_palm": ~self.palm_ever,
            "liftoff": self.liftoff,
            "max_robot_table_force": self.max_robot_table_force,
            "max_descent_tilt_deg": self.max_descent_tilt,
            "push_before_liftoff_m": self.push_before_liftoff,
        }


def summarize(outcomes: dict) -> dict:
    """Rates with Wilson 95 % CIs + median final distance."""
    n = int(outcomes["success"].numel())
    out: dict = {"episodes": n}
    keys = ("success", "success_5cm", "lift", "slam", "jam", "invalid", "clean", "clean_c1_no_table_hit",
            "clean_c2_top_down", "clean_c3_both_pads_at_liftoff", "clean_c4_no_palm", "liftoff", "table_contact",
            "reached_hover", "reached_descend", "reached_close", "reached_lift", "reached_carry", "reached_hold")
    for key in tuple(k for k in keys if k in outcomes) + tuple(
        k for k in outcomes if k.startswith("slam_") and k.endswith("N")
    ):
        k = int(outcomes[key].sum().item())
        lo, hi = wilson_ci(k, n)
        out[key] = {"count": k, "rate": k / n, "ci95": [lo, hi]}
    if "final_phase" in outcomes:
        out["final_phase_counts"] = [int((outcomes["final_phase"] == k).sum()) for k in range(6)]
        out["episodes_with_drop"] = int((outcomes["drops"] > 0).sum())
    d = outcomes["dist"]
    out["final_dist_median_m"] = float(torch.quantile(d, 0.5))
    out["final_dist_quartiles_m"] = [float(torch.quantile(d, q)) for q in (0.25, 0.75)]
    out["slam_or_jam"] = {"count": int((outcomes["slam"] | outcomes["jam"]).sum())}
    out["slam_or_jam"]["rate"] = out["slam_or_jam"]["count"] / n
    out["slam_or_jam"]["ci95"] = list(wilson_ci(out["slam_or_jam"]["count"], n))
    if "push_before_liftoff_m" in outcomes:
        lo = outcomes["liftoff"]
        pb = outcomes["push_before_liftoff_m"][lo] if bool(lo.any()) else outcomes["push_before_liftoff_m"]
        out["push_before_liftoff_median_m"] = float(torch.quantile(pb, 0.5))
    return out
