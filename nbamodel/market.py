"""Market-implied ratings: team power ratings solved from earlier closing lines.

A sharp book's numbers start from the market's own view of every team. For each game day, every
earlier closing spread (this season, plus last season at a lower weight, recency-weighted) is fit
as

    close_spread = hca * site + R_home - R_away - k * (missing_home - missing_away)

so each team's R is its full-strength rating as the market priced it, with the players who sat
out of each earlier game added back (k: points per unit of missing Game Score value, the same
units as availability.py). Totals the same way: close_total = mu + T_home + T_away.

The game model then uses R_home - R_away (and T_home + T_away) next to the box-score ratings, and
its own missing-players term handles tonight's lineups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HALF_LIFE = 30.0     # days (tried 15, 30, 60, 120 on 2022-2026: 30 was best)
RIDGE = 3.0          # game-weights toward an average team
PREV_WEIGHT = 0.5    # last season's lines, before the recency decay
K_MISS = 0.33        # points per unit of missing value, about what the game model learns
MIN_THIS_SEASON = 150  # lines from this season needed before the ratings are used (about 2 weeks)


def solve(h: pd.DataFrame, day: pd.Timestamp, season: int):
    """Ratings from the lines in `h` (all before `day`). Returns ({team: R}, hca, mu, {team: T})."""
    teams = sorted(set(h.home_id) | set(h.away_id))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (day - h.date).dt.days.to_numpy(float)
    w = 0.5 ** (age / HALF_LIFE) * np.where(h.season == season, 1.0, PREV_WEIGHT)
    m = len(h)
    r = np.arange(m)
    hi, ai = h.home_id.map(idx).to_numpy(), h.away_id.map(idx).to_numpy()
    X = np.zeros((m, n + 1))
    X[r, hi] = 1
    X[r, ai] = -1
    X[:, n] = np.where(h.neutral, 0, 1)
    miss_h = h.miss_h.fillna(0) if "miss_h" in h else 0.0
    miss_a = h.miss_a.fillna(0) if "miss_a" in h else 0.0
    y = h.close_spread.to_numpy(float) + K_MISS * np.asarray(miss_h - miss_a, dtype=float)
    A = X.T @ (X * w[:, None])
    pen = np.full(n + 1, RIDGE)
    pen[n] = 0.01
    A[np.diag_indices_from(A)] += pen
    b = np.linalg.solve(A, X.T @ (w * y))
    T = np.zeros((m, n))
    T[r, hi] = 1
    T[r, ai] += 1
    yt = h.close_total.to_numpy(float)
    mu = float(np.average(yt, weights=w))
    At = T.T @ (T * w[:, None])
    At[np.diag_indices_from(At)] += RIDGE
    bt = np.linalg.solve(At, T.T @ (w * (yt - mu)))
    return ({t: float(b[i]) for t, i in idx.items()}, float(b[n]), mu, {t: float(bt[i]) for t, i in idx.items()})


def market_ratings(df: pd.DataFrame, seasons=None) -> pd.DataFrame:
    """Pre-game market ratings for every game in `seasons` of `df` (needs close_spread,
    close_total, miss_h, miss_a, neutral). Returns game_id, mk_margin, mk_total."""
    df = df.sort_values(["date", "game_id"])
    lines = df.dropna(subset=["close_spread", "close_total"])
    seasons = seasons or sorted(df.season.unique())
    out = []
    for s in seasons:
        sd = df[df.season == s]
        pool = lines[lines.season.isin([s - 1, s])]
        for day, todays in sd.groupby("date"):
            h = pool[pool.date < day]
            if (h.season == s).sum() < MIN_THIS_SEASON:
                continue
            R, hca, mu, T = solve(h, day, s)
            for x in todays.itertuples():
                if x.home_id in R and x.away_id in R:
                    out.append((x.game_id, R[x.home_id] - R[x.away_id] + (0.0 if x.neutral else hca),
                                mu + T[x.home_id] + T[x.away_id]))
    return pd.DataFrame(out, columns=["game_id", "mk_margin", "mk_total"])
