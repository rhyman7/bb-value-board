import numpy as np
import pandas as pd

from nbamodel import data, model, ratings


def _games():
    # three teams, A clearly best, C worst, alternating home court
    rows = []
    day = pd.Timestamp("2026-11-01")
    pairs = [("A", "B", 115, 100), ("B", "C", 112, 104), ("C", "A", 98, 118)] * 6
    for i, (h, a, ph, pa) in enumerate(pairs):
        rows.append({"game_id": str(i), "season": 2027, "date": day + pd.Timedelta(days=i), "home_id": h, "away_id": a,
                     "home": h, "away": a, "pts_h": ph, "pts_a": pa, "poss": 100.0, "neutral": False})
    g = pd.DataFrame(rows)
    g["margin"] = g.pts_h - g.pts_a
    g["total"] = g.pts_h + g.pts_a
    return g


def test_ratings_order_teams():
    g = _games()
    R, mean, hca, pace = ratings.rate_day(g, pd.Timestamp("2026-12-01"), 2027, ratings.Settings())
    net = {t: o + d for t, (o, d, p) in R.items()}
    assert net["A"] > net["B"] > net["C"]


def test_rest_counts_back_to_backs():
    g = _games()
    g = data.add_rest(g)
    # every team plays every 1-2 days in the toy schedule; the first game of a season counts as rested
    assert g.rest_h.min() >= 0 and g.rest_h.max() <= 4
    assert (g.iloc[0][["rest_h", "rest_a"]] == 4).all()


def test_market_weight_recovers_signal():
    rng = np.random.default_rng(0)
    line = rng.normal(0, 6, 5000)
    model_ = line + rng.normal(0, 3, 5000)
    actual = line + 0.4 * (model_ - line) + rng.normal(0, 11, 5000)
    df = pd.DataFrame({"m": model_, "l": line, "a": actual})
    w, se, n = model.market_weight(df, "m", "l", "a")
    assert abs(w - 0.4) < 3 * se
