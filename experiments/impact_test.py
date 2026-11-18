"""Does an impact-based lineup strength help the NBA game model? Keep rule: better on the tuning
seasons (2017-20) and the holdout (2021-26), by at least one standard error on the holdout.

    python -m experiments.impact_test        # writes docs/nba/impact_test.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from nbamodel import backtest, impact, model

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "impact_test.json"
SEASONS = list(range(2016, 2027))
TUNE, HOLD = (2017, 2020), (2021, 2026)


def run(extra=None, feats=None, drop=None, avail=True):
    base_m = list(model.MARGIN_FEATURES)
    try:
        model.MARGIN_FEATURES[:] = [f for f in base_m if f not in (drop or [])] + (feats or [])
        bt = backtest.run(2026, verbose=False, use_availability=avail, extra=extra)["games"]
    finally:
        model.MARGIN_FEATURES[:] = base_m
    bt = bt.dropna(subset=["m_margin", "margin"])
    return bt.set_index("game_id")


def compare(base, new):
    j = base[["season", "margin", "m_margin"]].join(new[["m_margin"]], rsuffix="_new", how="inner")
    out = {}
    for name, (a, b) in (("tune", TUNE), ("holdout", HOLD)):
        x = j[(j.season >= a) & (j.season <= b)]
        d = (x.margin - x.m_margin_new).abs() - (x.margin - x.m_margin).abs()
        out[name] = {"games": int(len(x)), "base_mae": round(float((x.margin - x.m_margin).abs().mean()), 4),
                     "new_mae": round(float((x.margin - x.m_margin_new).abs().mean()), 4),
                     "change": round(float(d.mean()), 4), "se": round(float(d.std() / np.sqrt(len(d))), 4)}
    out["passes"] = out["tune"]["change"] < 0 and out["holdout"]["change"] < -out["holdout"]["se"]
    return out


if __name__ == "__main__":
    lams = [float(x) for x in sys.argv[1:]] or [1.0, 3.0, 10.0]
    res = {"note": "imp_diff = home lineup strength minus away, from walk-forward impact ratings"}
    for avail in (True, False):
        tag = "with_missing_players" if avail else "without_injury_hindsight"
        base = run(avail=avail)
        for lam in lams:
            fits, index = impact.walk(SEASONS, lam=lam)
            gs = impact.game_strength(SEASONS, fits, index)[["game_id", "imp_diff"]].fillna(0.0)
            res[f"{tag}_lam{lam:g}_add"] = r = compare(base, run(gs, ["imp_diff"], avail=avail))
            print(tag, lam, "add", json.dumps(r), flush=True)
            if avail:
                res[f"{tag}_lam{lam:g}_replace_missing"] = r = compare(base, run(gs, ["imp_diff"], drop=["miss_diff"], avail=avail))
                print(tag, lam, "replace", json.dumps(r), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


def prior_test(lams=(0.3, 1.0, 3.0)):
    """The same test with a box-score prior: compared with the impact ratings now in the model."""
    from nbamodel import impact as I
    base = run()
    out = {}
    for lam in lams:
        fits, index = I.walk(SEASONS, lam=lam, box_prior=True)
        gs = I.game_strength(SEASONS, fits, index)[["game_id", "imp_diff"]].rename(columns={"imp_diff": "imp_prior"})
        out[f"lam{lam:g}"] = r = compare(base, run(gs, ["imp_prior"], drop=["imp_diff"]))
        print("box prior", lam, json.dumps(r), flush=True)
    return out
