"""Plot training curves from rsl_rl TensorBoard logs.

  uv run python scripts/plot_curves.py --runs logs/rsl_rl/lift_rl/<run_a> logs/rsl_rl/lift_rl/<run_b> \
      --labels "M0 stock" "M1 own" --out results/figures/training_curves.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PANELS = [
    ("Train/mean_reward", "mean episode reward"),
    ("Metrics/success_rate", "success (training, sticky, see note)"),
    ("Episode_Termination/object_dropping", "episodes ending with cube dropped"),
    ("Metrics/object_pose/position_error", "hand-to-target distance [m]"),
]


def load(run: str) -> dict[str, tuple[list[int], list[float]]]:
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        ev = ea.Scalars(tag)
        out[tag] = ([e.step for e in ev], [e.value for e in ev])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    data = [load(r) for r in a.runs]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (tag, title) in zip(axes.flat, PANELS):
        for d, lab in zip(data, a.labels):
            if tag in d:
                ax.plot(*d[tag], label=lab, lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PPO iteration")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.text(0.01, 0.005, "Training success = cube ever within the threshold of the target while lifted "
             "(sticky, stochastic actions); the headline number is the 1,000-episode eval.", fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print(f"saved {a.out}; tags available: {sorted(data[0])[:40]}")


if __name__ == "__main__":
    main()
