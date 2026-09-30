"""Phase 7 figures (CPU only):
  results/figures/chain_test_results.png  test success / clean / table contact per phase (2-6), with 95 % CIs
  results/figures/learning_curves.png     training-time HOLD fraction and median cube-target distance vs iteration,
                                          phase 2 (run 5) and the fine-tunes of phases 3-6
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TESTS = [("2 fixed", "s16_run5_test.json"), ("3 +position", "p3_test.json"), ("4 +yaw", "p4_test.json"),
         ("5 +target", "p5_test.json"), ("6 +mass/friction", "p6_test.json")]
LOGS = [("phase 2 (from scratch, demo resets)", "s16_run5r_train.log"), ("phase 3", "p3_a1_train.log"),
        ("phase 4", "p4_a1_train.log"), ("phase 5", "p5_a1_train.log"), ("phase 6", "p6_a1_train.log")]

# ---- chain test results
rows = [(n, json.loads((ROOT / "results" / f).read_text())) for n, f in TESTS if (ROOT / "results" / f).exists()]
fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
x = np.arange(len(rows))
for k, (key, label, col) in enumerate((("success", "success (3 cm)", "#2a9d8f"), ("clean", "clean grasp", "#457b9d"),
                                       ("table_contact", "table contact", "#e76f51"))):
    v = np.array([100 * d[key]["rate"] for _, d in rows])
    lo = np.array([100 * d[key]["ci95"][0] for _, d in rows])
    hi = np.array([100 * d[key]["ci95"][1] for _, d in rows])
    ax.bar(x + (k - 1) * 0.27, v, 0.27, yerr=[v - lo, hi - v], capsize=3, label=label, color=col)
ax.axhline(90, color="k", lw=0.8, ls="--")
ax.text(len(rows) - 0.5, 91, "gate 90 %", ha="right", fontsize=8)
ax.set_xticks(x, [n for n, _ in rows])
ax.set_ylabel("% of test episodes")
ax.set_ylim(0, 105)
ax.set_title("Test results per phase (1,000 episodes; phase 2: 200), seed 12345")
ax.legend(loc="center right", fontsize=8)
fig.savefig(FIG / "chain_test_results.png", dpi=130)

# ---- learning curves from the training logs (rsl_rl console output)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for name, log in LOGS:
    f = ROOT / "logs" / "jobs" / log
    if not f.exists():
        continue
    txt = f.read_text(encoding="utf-8", errors="ignore")
    its, hold, dist = [], [], []
    for block in txt.split("Learning iteration ")[1:]:
        m = re.match(r"(\d+)/", block)
        h = re.search(r"Metrics/phase_6_frac:\s*([-\d.]+)", block)
        d = re.search(r"Metrics/cube_target_dist_median:\s*([-\d.]+)", block)
        if m and h:
            its.append(int(m.group(1)))
            hold.append(100 * float(h.group(1)))
            dist.append(100 * float(d.group(1)) if d else np.nan)
    if not its:
        continue
    axes[0].plot(its, hold, lw=1, label=name)
    axes[1].plot(its, dist, lw=1, label=name)
axes[0].set_xlabel("PPO iteration")
axes[0].set_ylabel("envs in HOLD [%] (training, mid-episode)")
axes[1].set_xlabel("PPO iteration")
axes[1].set_ylabel("median cube-target distance [cm]")
axes[1].set_yscale("log")
axes[0].legend(fontsize=7)
fig.suptitle("Training curves: phase 2 from scratch, phases 3-6 fine-tuned from the previous phase")
fig.savefig(FIG / "learning_curves.png", dpi=130)
print("wrote", [p.name for p in FIG.glob("*.png")])
