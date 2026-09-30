"""Section 17.2 twist comparison: phase 6 (trained with cube mass/friction randomization) vs phase 5 (without), both on
the full-variation world (cube position + yaw, random target; LiftRL-Phased-P5-TS-v0) with the cube mass and grasp
friction FIXED per cell: mass {0.1, 0.216, 0.35, 0.5} kg x friction {0.5, 0.8, 1.25}, 200 episodes per cell, seed 12345.
Each cell is one scripts/evaluate.py run (via ops/run_job.ps1). Writes results/twist_grid.json and
results/figures/twist_grid.png (success heatmaps).

Usage: python scripts/twist_grid.py --p5 <ckpt> --p6 <ckpt>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASSES = (0.1, 0.216, 0.35, 0.5)
FRICTIONS = (0.5, 0.8, 1.25)
TASK = "LiftRL-Phased-P5-TS-v0"

p = argparse.ArgumentParser()
p.add_argument("--p5", required=True)
p.add_argument("--p6", required=True)
p.add_argument("--episodes", type=int, default=200)
args = p.parse_args()

res = {"task": TASK, "episodes_per_cell": args.episodes, "seed": 12345, "masses": MASSES, "frictions": FRICTIONS,
       "policies": {"phase5": args.p5, "phase6": args.p6}, "cells": []}
for name, ck in (("phase5", args.p5), ("phase6", args.p6)):
    for m in MASSES:
        for f in FRICTIONS:
            out = ROOT / "results" / "twist_grid" / f"{name}_m{m}_f{f}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                job = f"grid_{name}_m{m}_f{f}"
                subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "ops" / "run_job.ps1"),
                                "-Job", job, "-LogFile", f"logs/jobs/{job}.log", "-PyArgs",
                                f"scripts/evaluate.py --task {TASK} --checkpoint {ck} --num_envs {args.episodes} "
                                f"--num_episodes {args.episodes} --seed 12345 --cube_mass {m} --cube_friction {f} --out {out}"],
                               cwd=ROOT, check=True)
            d = json.loads(out.read_text())
            res["cells"].append({"policy": name, "mass": m, "friction": f,
                                 **{k: d[k]["rate"] for k in ("success", "clean", "table_contact", "lift")},
                                 "success_ci95": d["success"]["ci95"], "applied": d.get("applied_physics"),
                                 "final_dist_median_m": d["final_dist_median_m"]})
            c = res["cells"][-1]
            print(f"{name} m={m} f={f}: success {c['success']:.3f} clean {c['clean']:.3f} applied {c['applied']}", flush=True)
(ROOT / "results" / "twist_grid.json").write_text(json.dumps(res, indent=1))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for ax, name in zip(axes, ("phase5", "phase6")):
    g = np.array([[next(c["success"] for c in res["cells"] if c["policy"] == name and c["mass"] == m and c["friction"] == f)
                   for f in FRICTIONS] for m in MASSES]) * 100
    im = ax.imshow(g, vmin=0, vmax=100, cmap="viridis", origin="lower")
    ax.set_xticks(range(len(FRICTIONS)), [str(f) for f in FRICTIONS])
    ax.set_yticks(range(len(MASSES)), [str(m) for m in MASSES])
    ax.set_xlabel("grasp friction")
    ax.set_ylabel("cube mass [kg]")
    ax.set_title(f"{'phase 5 (no twist)' if name == 'phase5' else 'phase 6 (twist)'}: success %")
    for i in range(len(MASSES)):
        for j in range(len(FRICTIONS)):
            ax.text(j, i, f"{g[i, j]:.0f}", ha="center", va="center", color="w" if g[i, j] < 60 else "k")
fig.colorbar(im, ax=axes, shrink=0.8)
(ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(ROOT / "results" / "figures" / "twist_grid.png", dpi=130)
print("wrote results/twist_grid.json and results/figures/twist_grid.png")
