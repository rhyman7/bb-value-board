"""Opponent-adjusted team ratings, walked forward one game day at a time.

For every day with games, each team's offense and defense (points per 100 possessions) and pace
(possessions per game) are solved by ridge regression over every earlier game: this season's
games, weighted by recency, plus last season's at a fraction of the weight. The same engine as the
football boards' EPA ratings, on possessions instead of plays.

    pts_per_100(team i vs j) = mean + off_i - def_j + hca * home
    poss(i vs j)             = mean + pace_i + pace_j

A team's rating on a day uses only games before that day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Settings:
    half_life: float = 45.0       # days: a game this many days old counts half (chosen on 2017-2020)
    last_season: float = 0.35     # weight on last season's games, before the recency decay
    last_season_decay: float = 0.5  # extra multiplier per 30 days into this season (fades as games pile up)
    ridge_eff: float = 6.0        # penalty on offense/defense ratings, in game-weights (chosen on 2017-2020)
    ridge_pace: float = 4.0
    hca_prior: float = 2.5        # points per 100 the home-court term shrinks toward
    hca_ridge: float = 40.0


def _solve(rows, n, y, w, ridge, extra_prior=None, extra_ridge=None, two_sided=True):
    """Weighted ridge for y = mean + X beta. Returns beta and the weighted mean."""
    mean = np.average(y, weights=w)
    X = rows
    W = w[:, None]
    A = X.T @ (X * W)
    pen = np.full(X.shape[1], ridge)
    prior = np.zeros(X.shape[1])
    if extra_prior is not None:
        pen[-1] = extra_ridge
        prior[-1] = extra_prior
    A[np.diag_indices_from(A)] += pen
    b = X.T @ (w * (y - mean)) + pen * prior
    return np.linalg.solve(A, b), mean


def rate_day(hist: pd.DataFrame, day: pd.Timestamp, season: int, s: Settings):
    """Ratings for every team from the games in `hist` (all before `day`).
    Returns {team_id: (off, def, pace)}, league eff mean, hca (per 100), pace mean."""
    if hist.empty:
        return {}, 110.0, s.hca_prior, 100.0
    teams = sorted(set(hist.home_id) | set(hist.away_id))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    age = (day - hist.date).dt.days.to_numpy(float)
    w = 0.5 ** (age / s.half_life)
    this = (hist.season == season).to_numpy()
    n_this_days = max(0.0, (day - hist.loc[this, "date"].min()).days) if this.any() else 0.0
    prev_w = s.last_season * (s.last_season_decay ** (n_this_days / 30.0))
    # last season's games: no recency decay among themselves beyond a flat weight
    w = np.where(this, w, prev_w)

    hi = hist.home_id.map(idx).to_numpy()
    ai = hist.away_id.map(idx).to_numpy()
    m = len(hist)
    neutral = hist.neutral.to_numpy()
    hsign = np.where(neutral, 0.0, 1.0)

    # efficiency: two rows per game (home offense, away offense)
    X = np.zeros((2 * m, 2 * n + 1))
    r = np.arange(m)
    X[r, hi] = 1; X[r, n + ai] = -1; X[r, 2 * n] = hsign
    X[m + r, ai] = 1; X[m + r, n + hi] = -1; X[m + r, 2 * n] = -hsign
    poss = hist.poss.to_numpy(float)
    y = np.concatenate([100 * hist.pts_h.to_numpy(float) / poss, 100 * hist.pts_a.to_numpy(float) / poss])
    ww = np.concatenate([w, w])
    beta, mean = _solve(X, n, y, ww, s.ridge_eff, extra_prior=s.hca_prior / 2, extra_ridge=s.hca_ridge)

    # pace
    P = np.zeros((m, n))
    P[r, hi] = 1; P[r, ai] += 1
    pbeta, pmean = _solve(P, n, poss, w, s.ridge_pace)

    out = {t: (float(beta[idx[t]]), float(beta[n + idx[t]]), float(pbeta[idx[t]])) for t in teams}
    return out, float(mean), float(beta[2 * n] * 2), float(pmean)


def walk_forward(g: pd.DataFrame, s: Settings | None = None, seasons=None) -> pd.DataFrame:
    """Pre-game ratings for every game in `seasons` (default: all but the first season in `g`)."""
    s = s or Settings()
    seasons = seasons or sorted(g.season.unique())[1:]
    rows = []
    for season in seasons:
        sg = g[g.season == season]
        pool = g[g.season.isin([season - 1, season])]
        for day, todays in sg.groupby("date"):
            hist = pool[pool.date < day]
            R, mean, hca, pmean = rate_day(hist, day, season, s)
            for x in todays.itertuples():
                rh = R.get(x.home_id, (0.0, 0.0, 0.0))
                ra = R.get(x.away_id, (0.0, 0.0, 0.0))
                rows.append({
                    "game_id": x.game_id,
                    "off_h": rh[0], "def_h": rh[1], "pace_h": rh[2],
                    "off_a": ra[0], "def_a": ra[1], "pace_a": ra[2],
                    "lg_eff": mean, "lg_hca": hca, "lg_pace": pmean,
                    "n_this": int((hist.season == season).sum()),
                })
    return g.merge(pd.DataFrame(rows), on="game_id")


def rating_projection(df: pd.DataFrame) -> pd.DataFrame:
    """The ratings' own projection: margin and total in points, before the game model."""
    df = df.copy()
    pace = df.lg_pace + df.pace_h + df.pace_a
    hca = np.where(df.neutral, 0.0, df.lg_hca)
    eff_h = df.lg_eff + df.off_h - df.def_a + hca / 2
    eff_a = df.lg_eff + df.off_a - df.def_h - hca / 2
    df["r_pace"] = pace
    df["r_margin"] = pace * (eff_h - eff_a) / 100
    df["r_total"] = pace * (eff_h + eff_a) / 100
    df["r_net_diff"] = (df.off_h + df.def_h) - (df.off_a + df.def_a)
    return df
