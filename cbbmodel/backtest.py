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

from . import availability, blend, data, lineup, ratings, roster, spots

SETTINGS = ratings.Settings(half_life=240.0, ridge_eff=3.0, carry=0.85, roster_k=0.3)
FEATURES = ["r_margin", "r_early", "rest_diff", "b2b_h", "b2b_a", "far_road_diff", "miss_diff", "lineup_diff"]
# miss_diff: home minus away value of regulars who have sat their team's last 2 games in a row
# (cbbmodel/availability.py; no college injury report). Added 2026-11-17 after it passed the keep rule:
# holdout 2019-23 home games 8.974 -> 8.969 (0.0052 +/- 0.0024), tuning 8.902 -> 8.890
# (experiments/cbb_missing.py, docs/cbb/missing_test.json).
# far_road_diff: road team 2+ time zones from home (spots.py). Added at Ryan's request on 2026-11-17
# as a small adjustment to watch: it pointed the same way on 2016-18 and 2019-23 (about half a point)
# but within one standard error, so it did not pass the keep rule on its own.
EARLY_GAMES = 3000      # Division I games played league-wide by about mid-December
FIRST_RATED = 2016
GRADE = [2019, 2020, 2021, 2022, 2023]
TUNE = [2016, 2017, 2018]
OUT = Path(__file__).resolve().parent.parent / "docs" / "cbb"


def fit(train):
    X = train[FEATURES].to_numpy(float)
    return dict(zip(FEATURES, np.linalg.solve(X.T @ X + np.eye(len(FEATURES)), X.T @ train.margin.to_numpy(float))))


def add_features(df):
    """The ratings run compressed early in the season (they lean on a shrunk preseason prior), so
    the model gets a second, fading copy of the ratings' margin to stretch them back out."""
    df = nba_model.features(df)
    df["r_early"] = df.r_margin * np.clip(1 - df.n_this / EARLY_GAMES, 0, 1)
    df["lineup_diff"] = df["lineup_diff"].fillna(0.0) if "lineup_diff" in df else 0.0
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
        "ratings_model_mae": round(float((x.margin - x.m_ratings).abs().mean()), 2) if "m_ratings" in x else None,
        "line_mae": round(float((x.margin - x.close_spread).abs().mean()), 2),
        "market_weight": round(w, 3), "market_weight_se": round(se, 3) if se else None,
        "ats_every_game": record(x),
        "ats_by_gap": {f"{k}+": record(x[(x.m_margin - x.close_spread).abs() >= k]) for k in [2, 3, 4, 5, 6]},
    }


def run(last_season, verbose=True, extra=None):
    """extra: optional per-game feature columns (with game_id) for experiments, filled with 0 when missing."""
    g = data.all_games(last_season, refresh_spreads=True)
    shifts = roster.prior_shift(range(FIRST_RATED, last_season + 1))
    seasons = list(range(FIRST_RATED, last_season + 1))
    rated = rating_projection(ratings.walk_forward(g, SETTINGS, seasons=seasons, shifts=shifts))
    rated = rated.merge(availability.missing_value(g, seasons)[["game_id", "miss_h", "miss_a"]], on="game_id", how="left")
    rated = rated.merge(lineup.history(g, seasons, current=last_season)[["game_id", "lineup_diff"]], on="game_id", how="left")
    rated = add_features(rated)
    if extra is not None:
        rated = rated.merge(extra, on="game_id", how="left")
        cols = [c for c in extra.columns if c != "game_id"]
        rated[cols] = rated[cols].fillna(0.0)
    # situational spots: each tested against the line on its own; only far_road_diff is in the model
    rated = rated.merge(spots.spots(g, rated), on="game_id", how="left")
    rated["far_road_diff"] = rated.far_road_diff.fillna(0.0)
    sp = rated
    spot_tests = {c: {"label": lab, "tune": spots.test(sp[sp.season.isin(TUNE)], c), "holdout": spots.test(sp[sp.season.isin(GRADE)], c),
                      "this_season": spots.test(sp[sp.season == last_season], c)} for c, (lab, _) in spots.TESTS.items()}
    out, coefs = [], {}
    for s in [x for x in seasons if x >= FIRST_RATED + 2]:
        coef = fit(rated[(rated.season < s) & (rated.season < last_season)])
        coefs[s] = coef
        out.append(predict(rated[rated.season == s], coef))
    bt = pd.concat(out, ignore_index=True)
    # the board's number: the ratings model averaged with two other formulas (blend.py), each
    # season's fitted on earlier seasons only
    hist, _ = blend.history(bt, FIRST_RATED, last_season)
    parts = [blend.predict(hist[hist.season == s], blend.fit(hist[(hist.season < s) & (hist.season < last_season)]))
             if s > FIRST_RATED + 2 else hist[hist.season == s].assign(m_ratings=lambda d: d.m_margin)
             for s in sorted(hist.season.unique())]
    bt = pd.concat(parts, ignore_index=True)
    held = bt[bt.season.isin(GRADE)]
    cur = bt[bt.season == last_season]
    summary = {
        "settings": SETTINGS.__dict__,
        "holdout_seasons": f"{GRADE[0]}-{GRADE[-1]}",
        "home_games": grade(held[~held.neutral]),
        "neutral_games_line_unreliable": grade(held[held.neutral]),
        "by_season": {int(s): grade(x[~x.neutral]) for s, x in held.groupby("season")},
        "this_season_so_far": grade(cur[~cur.neutral]) if cur.close_spread.notna().sum() >= 200 else None,
        "spots": spot_tests,
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
