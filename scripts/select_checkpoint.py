"""Apply the pre-registered checkpoint rule (pre-registered 2026-09-29) to validation eval JSONs and record the choice
BEFORE the test eval runs.

Rule: among the candidates, the highest validation success (3 cm) among those with validation clean >= 80 % and
slam <= 10 %; if none qualifies, the highest validation clean rate (ties: higher success).

Usage: uv run python scripts/select_checkpoint.py --vals results/r2_full_val_1000.json results/r2_full_val_1250.json \
           results/r2_full_val_1500.json --out results/r2_full_checkpoint_choice.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def choose(vals: list[dict]) -> tuple[dict, str]:
    ok = [v for v in vals if v["clean"]["rate"] >= 0.80 and v["slam"]["rate"] <= 0.10]
    if ok:
        return max(ok, key=lambda v: v["success"]["rate"]), "max validation success among clean >= 80 % and slam <= 10 %"
    return max(vals, key=lambda v: (v["clean"]["rate"], v["success"]["rate"])), "none qualified: max validation clean (ties: success)"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vals", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    vals = [json.loads(Path(f).read_text()) | {"_file": f} for f in a.vals]
    best, why = choose(vals)
    rec = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "rule": why,
        "chosen_checkpoint": best["checkpoint"],
        "chosen_val_file": best["_file"],
        "candidates": [{"checkpoint": v["checkpoint"], "file": v["_file"], "success": v["success"]["rate"],
                        "clean": v["clean"]["rate"], "slam": v["slam"]["rate"], "lift": v["lift"]["rate"],
                        "jam": v["jam"]["rate"]} for v in vals],
    }
    Path(a.out).write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
