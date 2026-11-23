"""Impact ratings from real lineups (play-by-play stints, nbamodel/pbp_impact.py) against the
current ones estimated from minutes. Same walk-forward fit (impact.walk), same lineup strength
(impact.game_strength), same keep rule as experiments/impact_test.py.

    python -m experiments.pbp_impact_test        # writes docs/nba/pbp_impact_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from experiments.impact_test import SEASONS, compare, run
from nbamodel import impact, pbp_impact

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "pbp_impact_test.json"


def _accumulate(rows, index, n):
    """Stint rows: variance of a stint's margin grows with its length, so weight 1 / minutes."""
    A = np.zeros((n, n))
    b = np.zeros(n)
    for (_, _, _, ids, coef, hc, y, m) in rows:
        w = 1.0 / max(m, 0.5)
        cols = np.array([index[a] for a in ids] + [n - 1])
        x = np.append(coef, hc)
        A[np.ix_(cols, cols)] += w * np.outer(x, x)
        b[cols] += w * x * y
    return A, b


def pbp_strength(lam):
    orig_rows, orig_acc = impact.season_rows, impact._accumulate
    impact.season_rows = lambda s, refresh=False: pbp_impact.season_stints(s, refresh)
    impact._accumulate = _accumulate
    try:
        fits, index = impact.walk(SEASONS, lam=lam)
    finally:
        impact.season_rows, impact._accumulate = orig_rows, orig_acc
    gs = impact.game_strength(SEASONS, fits, index)[["game_id", "imp_diff"]].rename(columns={"imp_diff": "pbp_diff"})
    return gs.fillna(0.0), fits, index


if __name__ == "__main__":
    lams = [float(x) for x in sys.argv[1:]] or [0.3, 1.0, 3.0]
    res = {"note": "pbp_diff = lineup strength from play-by-play stint ratings; base model already has imp_diff (minutes-estimated)"}
    base = run()
    for lam in lams:
        gs, fits, index = pbp_strength(lam)
        res[f"lam{lam:g}_replace_imp"] = r = compare(base, run(gs, ["pbp_diff"], drop=["imp_diff"]))
        print(lam, "replace", json.dumps(r), flush=True)
        res[f"lam{lam:g}_add"] = r = compare(base, run(gs, ["pbp_diff"]))
        print(lam, "add", json.dumps(r), flush=True)
        # top players by the last fit, as a sanity check
        last = fits[-1][1]
        inv = {i: a for a, i in index.items()}
        top = np.argsort(-last)[:10]
        res[f"lam{lam:g}_top_ids"] = [[int(inv[i]), round(float(last[i]), 2)] for i in top]
    OUT.write_text(json.dumps(res, indent=1))
