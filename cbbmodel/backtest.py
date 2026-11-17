"""Walk-forward backtest of the college model, graded against ESPN's spread.

Ratings use only games before each game's date; the game model for each season is fitted on
earlier seasons only. ESPN's spread is in hoopR's play-by-play for 2015-2023 and for this season;
the 2023-24 to 2025-26 files carry none. Settings were chosen on 2016-2018, so 2019 on is the
holdout.

ESPN's neutral-site spreads are sometimes attached to the wrong team (a 10-point favorite listed
as a 10-point underdog), which makes any model look good on neutral floors. Those games are
graded separately and the headline numbers are home-court games only.

    python -m cbbmodel.backtest        # writes docs/cbb/backtest.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nbamodel import model as nba_model
from nbamodel.ratings import rating_projection

from . import data, ratings, roster

SETTINGS = ratings.Settings(half_life=240.0, ridge_eff=3.0, carry=0.85, roster_k=0.3)
FEATURES = ["r_margin", "r_early", "rest_diff", "b2b_h", "b2b_a"]
EARLY_GAMES = 3000      # Division I games played league-wide by about mid-December
FIRST_RATED = 2016
GRADE = [2019, 2020, 2021, 2022, 2023]
OUT = Path(__file__).resolve().parent.parent / "docs" / "cbb"


def fit(train):
    X = train[FEATURES].to_numpy(float)
    return dict(zip(FEATURES, np.linalg.solve(X.T @ X + np.eye(len(FEATURES)), X.T @ train.margin.to_numpy(float))))


def add_features(df):
    """The ratings run compressed early in the season (they lean on a shrunk preseason prior), so
    the model gets a second, fading copy of the ratings' margin to stretch them back out."""
    df = nba_model.features(df)
    df["r_early"] = df.r_margin * np.clip(1 - df.n_this / EARLY_GAMES, 0, 1)
    return df


def predict(df, coef):
    df = df.copy()
    df["m_margin"] = sum(df[k] * v for k, v in coef.items())
    return df


def record(sub):
    res = np.sign(sub.margin - sub.close_spread) * np.sign(sub.m_margin - sub.close_spread)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    return {"w": w, "l": l, "p": p, "pct": round(100 * w / n, 1) if n else None,
            "roi": round(100 * (w * 100 / 110 - l) / max(1, w + l + p), 1), "se": round(100 * math.sqrt(0.25 / n), 1) if n else None}


def grade(x):
    x = x.dropna(subset=["close_spread"])
    w, se, n = nba_model.market_weight(x, "m_margin", "close_spread", "margin")
    return {
        "games": int(len(x)),
        "model_mae": round(float((x.margin - x.m_margin).abs().mean()), 2),
        "ratings_only_mae": round(float((x.margin - x.r_margin).abs().mean()), 2),
        "line_mae": round(float((x.margin - x.close_spread).abs().mean()), 2),
        "market_weight": round(w, 3), "market_weight_se": round(se, 3) if se else None,
        "ats_every_game": record(x),
        "ats_by_gap": {f"{k}+": record(x[(x.m_margin - x.close_spread).abs() >= k]) for k in [2, 3, 4, 5, 6]},
    }


def run(last_season, verbose=True):
    g = data.all_games(last_season, refresh_spreads=True)
    shifts = roster.prior_shift(range(FIRST_RATED, last_season + 1))
    seasons = list(range(FIRST_RATED, last_season + 1))
    rated = rating_projection(ratings.walk_forward(g, SETTINGS, seasons=seasons, shifts=shifts))
    rated = add_features(rated)
    out, coefs = [], {}
    for s in [x for x in seasons if x >= FIRST_RATED + 2]:
        coef = fit(rated[(rated.season < s) & (rated.season < last_season)])
        coefs[s] = coef
        out.append(predict(rated[rated.season == s], coef))
    bt = pd.concat(out, ignore_index=True)
    held = bt[bt.season.isin(GRADE)]
    cur = bt[bt.season == last_season]
    summary = {
        "settings": SETTINGS.__dict__,
        "holdout_seasons": f"{GRADE[0]}-{GRADE[-1]}",
        "home_games": grade(held[~held.neutral]),
        "neutral_games_line_unreliable": grade(held[held.neutral]),
        "by_season": {int(s): grade(x[~x.neutral]) for s, x in held.groupby("season")},
        "this_season_so_far": grade(cur[~cur.neutral]) if cur.close_spread.notna().sum() >= 200 else None,
        "coefficients": {str(k): {a: round(b, 3) for a, b in v.items()} for k, v in coefs.items()},
    }
    if verbose:
        print(json.dumps({k: summary[k] for k in ["home_games", "neutral_games_line_unreliable", "this_season_so_far"]}, indent=1))
    return {"summary": summary, "games": bt}


def main():
    from datetime import date
    today = date.today()
    last = today.year + 1 if today.month >= 8 else today.year
    res = run(last)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backtest.json").write_text(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
