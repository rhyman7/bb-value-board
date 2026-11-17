import numpy as np
import pandas as pd

from cbbmodel import ratings as cbb_ratings
from nbamodel import market


def _games(season=2027, n_rounds=6):
    rows = []
    day = pd.Timestamp("2026-11-01")
    pairs = [("A", "B", 80, 65), ("B", "C", 75, 70), ("C", "A", 60, 82)] * n_rounds
    for i, (h, a, ph, pa) in enumerate(pairs):
        rows.append({"game_id": f"{season}-{i}", "season": season, "date": day + pd.Timedelta(days=i), "home_id": h, "away_id": a,
                     "pts_h": ph, "pts_a": pa, "poss": 68.0, "neutral": False})
    g = pd.DataFrame(rows)
    g["margin"] = g.pts_h - g.pts_a
    return g


def test_college_ratings_order_teams_and_carry_prior():
    g = pd.concat([_games(2026, 15), _games(2027, 12)], ignore_index=True)
    r, final = cbb_ratings.walk_forward(g, cbb_ratings.Settings(), seasons=[2027], return_final=True)
    net = {t: final[t][0] + final[t][1] for t in "ABC"}
    assert net["A"] > net["B"] > net["C"]
    # the first game of 2027 already knows A is good, from last season
    first = r.sort_values("date").iloc[0]
    assert (first.off_h + first.def_h) > (first.off_a + first.def_a)


def test_market_ratings_recover_spreads():
    g = _games(2027, 80)
    g["close_spread"] = np.where(g.home_id == "A", 12.0, np.where(g.home_id == "B", 4.0, -10.0))
    g["close_total"] = 140.0
    g["miss_h"] = g["miss_a"] = 0.0
    R, hca, mu, T = market.solve(g, g.date.max() + pd.Timedelta(days=1), 2027)
    assert R["A"] > R["B"] > R["C"]
    assert abs(mu - 140.0) < 1e-6
