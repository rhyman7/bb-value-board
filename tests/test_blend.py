"""College blend: inputs for unplayed games come from the same state the history uses."""
import numpy as np
import pandas as pd

from cbbmodel import blend
from nbamodel import formulas as F


def _games():
    rows, d = [], pd.Timestamp("2024-11-05")
    rng = np.random.default_rng(0)
    for i in range(60):
        h, a = (1, 2) if i % 2 else (2, 3)
        rows.append({"game_id": str(i), "season": 2025, "date": d + pd.Timedelta(days=i), "home_id": h, "away_id": a,
                     "neutral": i % 5 == 0, "margin": float(rng.normal(3 if h == 1 else -2, 8))})
    return pd.DataFrame(rows)


def test_elo_state_matches_next_game():
    g = _games()
    s, r, last = F.elo(g, return_state=True)
    nxt = pd.DataFrame([{"game_id": "x", "season": 2025, "date": g.date.max() + pd.Timedelta(days=1), "home_id": 1,
                         "away_id": 2, "neutral": False, "margin": np.nan}])
    s2 = F.elo(pd.concat([g, nxt], ignore_index=True))
    assert last == 2025
    assert abs(s2["x"] - (r[1] - r[2] + F.ELO_HOME)) < 1e-9


def test_today_and_predict_fall_back_without_inputs():
    cols = [f"p_{c}_{k}" for c in ("efg", "tov", "orb", "ftr") for k in ("o", "d")] + ["p_mov", "p_pace"]
    state = {"form": pd.DataFrame(columns=cols, dtype=float), "elo": {1: 1600.0, 2: 1500.0}, "elo_season": 2025}
    row = pd.DataFrame([{"home_id": 1, "away_id": 2, "neutral": True, "m_margin": 4.0, "r_margin": 4.0, "r_early": 0.0,
                         "rest_diff": 0, "b2b_h": 0, "b2b_a": 0, "far_road_diff": 0.0, "miss_diff": 0.0, "lineup_diff": 0.0}])
    t = blend.today(row, state, 2026)
    assert abs(t.elo_diff.iloc[0] - 0.75 * 100) < 1e-9        # new season: carried at 75%, neutral floor
    out = blend.predict(t, (lambda X: np.full(len(X), 99.0), None))
    assert out.m_margin.iloc[0] == 4.0 and out.m_ratings.iloc[0] == 4.0     # no four factors yet: ratings model alone
