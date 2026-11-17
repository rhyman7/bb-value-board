"""Do Ryan's situational spots beat ESPN's line? For each spot, regress the result against the
line (margin - spread, home side) on the spot flag (home minus away), on the tuning seasons
(2016-2018), the holdout (2019-2023) and this season. Then add the flags to the college game model
and compare its miss and market weight.

    python -m experiments.cbb_spots
"""
import json
import numpy as np
import pandas as pd

from cbbmodel import backtest, data, ratings, roster, spots
from nbamodel import model as nba_model
from nbamodel.ratings import rating_projection

TUNE, HOLD, CUR = [2016, 2017, 2018], backtest.GRADE, [2027]


def load():
    g = data.all_games(2027, refresh_current=False)
    shifts = roster.prior_shift(range(backtest.FIRST_RATED, 2028))
    rated = rating_projection(ratings.walk_forward(g, backtest.SETTINGS, seasons=list(range(backtest.FIRST_RATED, 2028)), shifts=shifts))
    sp = spots.spots(g, rated)
    return g, backtest.add_features(rated).merge(sp, on="game_id", how="left")


def model_test(rated, extra):
    feats = backtest.FEATURES + extra
    out = []
    for s in sorted(rated.season.unique()):
        if s < backtest.FIRST_RATED + 2:
            continue
        tr = rated[(rated.season < s) & (rated.season < 2027)]
        X = tr[feats].to_numpy(float)
        coef = np.linalg.solve(X.T @ X + np.eye(len(feats)), X.T @ tr.margin.to_numpy(float))
        te = rated[rated.season == s].copy()
        te["m_margin"] = te[feats].to_numpy(float) @ coef
        out.append(te)
    bt = pd.concat(out)
    res = {}
    for name, ss in [("tune", [2018]), ("holdout", HOLD), ("this_season", CUR)]:
        x = bt[bt.season.isin(ss) & ~bt.neutral].dropna(subset=["close_spread"])
        w, se, _ = nba_model.market_weight(x, "m_margin", "close_spread", "margin")
        res[name] = {"mae": round(float((x.margin - x.m_margin).abs().mean()), 3), "weight": round(w, 3), "weight_se": round(se or 0, 3)}
    return res, dict(zip(feats, np.round(coef, 3)))


def main():
    g, rated = load()
    rated.to_parquet("data/cache/exp/cbb_spots_rated.parquet")
    cols = list(spots.TESTS)
    report = {}
    for c in cols:
        report[c] = {nm: spots.test(rated[rated.season.isin(ss)], c) for nm, ss in [("tune", TUNE), ("holdout", HOLD), ("this_season", CUR)]}
    print(json.dumps(report, indent=1))
    print("rival pairs:", int(rated.rival.sum()), "rival games")
    base, _ = model_test(rated, [])
    print("base", base)
    for c in cols:
        r, coef = model_test(rated, [c])
        print(c, r, coef.get(c))
    r, coef = model_test(rated, cols)
    print("all", r, {k: coef[k] for k in cols})


if __name__ == "__main__":
    main()
