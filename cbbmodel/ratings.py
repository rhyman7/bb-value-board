"""Opponent-adjusted college ratings, walked forward one game day at a time.

The same model as the NBA ratings (nbamodel/ratings.py), per 100 possessions:

    pts_per_100(team i vs j) = mean + off_i - def_j + hca * site      (site: +1 home, -1 away, 0 neutral)
    poss(i vs j)             = mean + pace_i + pace_j

solved by weighted ridge over every earlier game, with three college changes:

- a preseason prior: each team starts the season at its final rating from last season, pulled
  toward the league average (`carry`), and the ridge shrinks toward that prior instead of zero.
  365 teams play ~30 games each, so in November the prior carries most of the weight;
- the normal equations are built directly (np.add.at), because 365 teams x 6,000 games a season
  makes the NBA engine's dense design matrix slow;
- recency weighting within the season is mild (college teams change less within a season than
  between them).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Settings:
    half_life: float = 120.0     # days
    ridge_eff: float = 3.0       # game-weights pulling each rating toward its prior
    ridge_pace: float = 3.0
    carry: float = 0.65          # share of last season's final rating kept as this season's prior
    hca: float = 5.0             # home court, points per 100 possessions (about 3.5 points)
    hca_ridge: float = 2000.0    # the home-court term barely moves off `hca` within a season
    roster_k: float = 0.0        # per-100 points of prior per unit of roster shift (roster.py)


def _solve_eff(hi, ai, site, y, w, n, prior_off, prior_def, s):
    """Ridge with prior: rows (off_i, def_j, hca). Returns off, def, hca, mean."""
    mean = np.average(y, weights=w)
    p = 2 * n + 1
    A = np.zeros((p, p))
    b = np.zeros(p)
    # each observation: x = e_off(i) - e_def(j) + site * e_hca
    cols = [(hi, 1.0), (n + ai, -1.0), (np.full(len(hi), 2 * n), site)]
    r = y - mean
    for ci, si in cols:
        np.add.at(b, ci, w * si * r)
        for cj, sj in cols:
            np.add.at(A, (ci, cj), w * si * sj)
    pen = np.concatenate([np.full(2 * n, s.ridge_eff), [s.hca_ridge]])
    prior = np.concatenate([prior_off, prior_def, [s.hca / 2]])
    A[np.diag_indices(p)] += pen
    beta = np.linalg.solve(A, b + pen * prior)
    return beta[:n], beta[n:2 * n], beta[2 * n] * 2, mean


def _solve_pace(hi, ai, poss, w, n, prior_pace, s):
    mean = np.average(poss, weights=w)
    A = np.zeros((n, n))
    b = np.zeros(n)
    r = poss - mean
    for ci in (hi, ai):
        np.add.at(b, ci, w * r)
        for cj in (hi, ai):
            np.add.at(A, (ci, cj), w)
    A[np.diag_indices(n)] += s.ridge_pace
    return np.linalg.solve(A, b + s.ridge_pace * prior_pace), mean


def walk_forward(g: pd.DataFrame, s: Settings | None = None, seasons=None, return_final=False, shifts=None):
    """Pre-game ratings for every game in `seasons`. Each season's prior is the previous season's
    final ratings times `carry`."""
    s = s or Settings()
    seasons = seasons or sorted(g.season.unique())
    rows = []
    final = {}
    for season in sorted(g.season.unique()):
        sg = g[g.season == season].copy()
        teams = sorted(set(sg.home_id) | set(sg.away_id))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        po = np.array([s.carry * final.get(t, (0, 0, 0))[0] for t in teams])
        pd_ = np.array([s.carry * final.get(t, (0, 0, 0))[1] for t in teams])
        pp = np.array([s.carry * final.get(t, (0, 0, 0))[2] for t in teams])
        if shifts and season in shifts and s.roster_k:
            sh = np.array([shifts[season].get(t, 0.0) for t in teams])
            sh = s.roster_k * (sh - sh.mean())
            po, pd_ = po + sh / 2, pd_ + sh / 2
        hi_all = sg.home_id.map(idx).to_numpy()
        ai_all = sg.away_id.map(idx).to_numpy()
        site_all = np.where(sg.neutral.to_numpy(), 0.0, 1.0)
        dates = sg.date.to_numpy()
        poss = sg.poss.to_numpy(float)
        ph = 100 * sg.pts_h.to_numpy(float) / poss
        pa = 100 * sg.pts_a.to_numpy(float) / poss
        mean_prev = (final.get("_mean", 104.0), final.get("_pace", 69.0))
        days = np.unique(dates)
        do_rate = season in seasons
        last = None
        for day in list(days) + [None]:
            mask = dates < day if day is not None else np.ones(len(dates), bool)
            k = int(mask.sum())
            if k < 30:
                off, dfn, hca, mean = po, pd_, s.hca, mean_prev[0]
                pace, pmean = pp, mean_prev[1]
            else:
                age = ((day if day is not None else dates.max()) - dates[mask]) / np.timedelta64(1, "D")
                w = 0.5 ** (age / s.half_life)
                hi, ai, site = hi_all[mask], ai_all[mask], site_all[mask]
                H = np.concatenate([hi, ai]); A_ = np.concatenate([ai, hi])
                S = np.concatenate([site, -site]); Y = np.concatenate([ph[mask], pa[mask]]); W = np.concatenate([w, w])
                off, dfn, hca, mean = _solve_eff(H, A_, S, Y, W, n, po, pd_, s)
                pace, pmean = _solve_pace(hi, ai, poss[mask], w, n, pp, s)
            if day is None:
                last = (off, dfn, pace, mean, pmean)
                break
            if not do_rate:
                continue
            today = np.where(dates == day)[0]
            for j in today:
                h, a = hi_all[j], ai_all[j]
                rows.append({"game_id": sg.game_id.iloc[j],
                             "off_h": off[h], "def_h": dfn[h], "pace_h": pace[h],
                             "off_a": off[a], "def_a": dfn[a], "pace_a": pace[a],
                             "lg_eff": mean, "lg_hca": hca, "lg_pace": pmean, "n_this": k})
        off, dfn, pace, mean, pmean = last
        final = {t: (off[idx[t]], dfn[idx[t]], pace[idx[t]]) for t in teams}
        final["_mean"], final["_pace"] = mean, pmean
    out = g[g.season.isin(seasons)].merge(pd.DataFrame(rows), on="game_id")
    return (out, final) if return_final else out
