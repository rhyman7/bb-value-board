"""Does knowing who is out help the college game model?

Adds miss_diff (home missing value minus away, Game Score units per game, nbamodel/availability.py
on college boxes) to the college model, two ways:
  inferred  only regulars who sat the team's last 2 games in a row (what the board can know)
  truth     every regular who did not play (an upper bound: a perfect injury report)
Graded like cbbmodel/backtest.py: fit on earlier seasons only, home-court games with ESPN's line,
2016-18 (tuning) and 2019-23 (holdout).

    python -m experiments.cbb_missing     # writes docs/cbb/missing_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbbmodel import availability, backtest as B, data, ratings, roster, spots
from nbamodel.ratings import rating_projection

OUT = Path(__file__).resolve().parent.parent / "docs" / "cbb" / "missing_test.json"


def run(last=2023):
    g = data.all_games(last, refresh_current=False)
    seasons = list(range(B.FIRST_RATED, last + 1))
    shifts = roster.prior_shift(seasons)
    rated = B.add_features(rating_projection(ratings.walk_forward(g, B.SETTINGS, seasons=seasons, shifts=shifts)))
    rated = rated.merge(spots.spots(g, rated), on="game_id", how="left")
    rated["far_road_diff"] = rated.far_road_diff.fillna(0.0)
    variants = {"base": None}
    for name, k in (("inferred", availability.INFER_K), ("truth", None)):
        mv = availability.missing_value(g, seasons, infer_k=k)
        rated = rated.merge(mv[["game_id", "miss_h", "miss_a"]].rename(columns={"miss_h": f"mh_{name}", "miss_a": f"ma_{name}"}), on="game_id", how="left")
        rated[f"miss_diff_{name}"] = (rated[f"mh_{name}"].fillna(0) - rated[f"ma_{name}"].fillna(0))
        variants[name] = f"miss_diff_{name}"
        print(name, "games with someone missing:", int((rated[f"miss_diff_{name}"] != 0).sum()), flush=True)
    res = {}
    preds = {}
    for name, col in variants.items():
        feats = B.FEATURES + ([col] if col else [])
        out = []
        for s in [x for x in seasons if x >= B.FIRST_RATED + 2]:
            tr = rated[rated.season < s]
            X = tr[feats].to_numpy(float)
            coef = dict(zip(feats, np.linalg.solve(X.T @ X + np.eye(len(feats)), X.T @ tr.margin.to_numpy(float))))
            te = rated[rated.season == s].copy()
            te["m_margin"] = sum(te[k] * v for k, v in coef.items())
            te["coef_miss"] = coef.get(col, np.nan) if col else np.nan
            out.append(te)
        bt = pd.concat(out, ignore_index=True)
        preds[name] = bt.set_index("game_id")
        res[name] = {"coef_last": round(float(bt.coef_miss.iloc[-1]), 3) if col else None}
    for name in variants:
        for per, ss in (("tune", B.TUNE), ("holdout", B.GRADE)):
            x = preds[name]
            x = x[x.season.isin(ss) & ~x.neutral].dropna(subset=["close_spread"])
            base = preds["base"].loc[x.index]
            d = (x.margin - x.m_margin).abs() - (base.margin - base.m_margin).abs()
            res[name][per] = {"games": int(len(x)), "mae": round(float((x.margin - x.m_margin).abs().mean()), 3),
                              "vs_base": round(float(d.mean()), 4), "vs_base_se": round(float(d.std() / np.sqrt(len(d))), 4),
                              "line_mae": round(float((x.margin - x.close_spread).abs().mean()), 3),
                              "ats": B.record(x)}
            sub = x[x[variants[name]] != 0] if variants[name] else None
            if sub is not None and len(sub):
                bs = preds["base"].loc[sub.index]
                res[name][per]["missing_games"] = {"games": int(len(sub)), "mae": round(float((sub.margin - sub.m_margin).abs().mean()), 3),
                                                   "base_mae": round(float((bs.margin - bs.m_margin).abs().mean()), 3),
                                                   "line_mae": round(float((sub.margin - sub.close_spread).abs().mean()), 3)}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    run()
