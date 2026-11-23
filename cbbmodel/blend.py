"""The college model's number: the average of three formulas on the same data.

1. The ratings model (backtest.fit: ratings plus rest, travel, missing players and lineup).
2. One ridge regression over those inputs plus each team's four factors (shooting, turnovers,
   offensive rebounds, free throws, for and against), recent point margin and Elo.
3. Gradient-boosted trees over the same inputs.

Tested walk-forward on home-court games against ESPN's spread (experiments/formulas_cbb.py,
docs/cbb/formula_test.json): the average missed the margin by 0.019 +/- 0.012 points less than the
ratings model alone on the tuning season (2019) and 0.023 +/- 0.005 less on 2020-23. Its record
against the spread is unchanged, about 50%. Added Nov 23, 2026 with Ryan's go-ahead.

A game whose teams have fewer than 5 earlier games (no four factors yet) keeps the ratings model's
number alone.
"""
from __future__ import annotations

import pandas as pd

from nbamodel import formulas as F

from . import data

BASE = ["home", "rest_diff", "b2b_h", "b2b_a", "far_road_diff", "miss_diff", "lineup_diff"]
COLS = ["r_margin", "r_early"] + BASE + F.FF + ["elo_diff", "form_diff"]
LAM = 10.0


def history(rated, first_season, last_season):
    """rated (one row per game, with margin) plus the blend's inputs before each game. Returns the
    frame and the state needed for games not yet played (current form, Elo)."""
    g = rated.drop(columns=[c for c in rated.columns if c.startswith("p_") or c in F.FF + ["form_diff", "elo_diff", "home"]])
    g["game_id"] = g.game_id.astype(str)
    g["home_id"], g["away_id"] = g.home_id.astype(int), g.away_id.astype(int)
    tg = F.team_games(range(first_season - 1, last_season + 1), data, types=(2, 3))
    g = F.attach(g, F.prior_form(tg))
    e, ratings, last = F.elo(g, return_state=True)
    g["elo_diff"] = g.game_id.map(e)
    g["home"] = (~g.neutral.astype(bool)).astype(float)
    return g, {"form": F.current_form(tg), "elo": ratings, "elo_season": last}


def fit(train):
    """The two extra formulas, fitted on games with every input."""
    t = train.dropna(subset=COLS + ["margin"])
    X, y = t[COLS].to_numpy(float), t.margin.to_numpy(float)
    lin = F.ridge(X, y, LAM)
    trees = F.gbm().fit(X, y)
    return lin, trees


def today(rows: pd.DataFrame, state, season):
    """Fill the blend's inputs for games not yet played (rows have home_id, away_id, neutral)."""
    g = rows.copy()
    form = state["form"]
    for side, idc in (("h", "home_id"), ("a", "away_id")):
        f = form.reindex(g[idc].astype(int).to_numpy())
        for c in form.columns:
            g[f"{c}_{side}"] = f[c].to_numpy()
    g = F.matchup(g)
    r = state["elo"]
    carry = state["elo_season"] is not None and state["elo_season"] < season
    rate = lambda t: F.ELO_START + (F.ELO_CARRY if carry else 1.0) * (r.get(int(t), F.ELO_START) - F.ELO_START)
    g["elo_diff"] = [rate(h) - rate(a) + (0 if bool(n) else F.ELO_HOME) for h, a, n in zip(g.home_id, g.away_id, g.neutral)]
    g["home"] = (~g.neutral.astype(bool)).astype(float)
    return g


def predict(g, models):
    """The average of the ratings model (m_margin) and the two extra formulas; m_margin alone where
    an input is missing. Adds m_ratings (the ratings model alone) and m_margin (the blend)."""
    lin, trees = models
    g = g.copy()
    g["m_ratings"] = g.m_margin
    ok = g[COLS].notna().all(axis=1).to_numpy()
    if ok.any():
        X = g.loc[ok, COLS].to_numpy(float)
        g.loc[ok, "m_margin"] = (g.loc[ok, "m_ratings"].to_numpy() + lin(X) + trees.predict(X)) / 3
    return g
