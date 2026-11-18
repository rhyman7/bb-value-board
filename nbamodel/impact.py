"""Player impact ratings: each player's effect on his team's scoring margin while he is on the
court, in points per 48 minutes, from box-score plus-minus (a ridge-regressed plus-minus, like RAPM).

There is no lineup data in the free box scores, so who shared the court is estimated from minutes:
a teammate who played 36 of 48 minutes is assumed on the floor for 36/48 of a player's minutes
(scaled so five are always on), and the same for opponents. Each player-game's plus-minus is then

    pm_i = (min_i / 48) * ( home court + sum of teammates' ratings x overlap - opponents' ratings x overlap )

with the player's own rating counted in full. Ratings are shrunk toward zero (an average player) by a
ridge penalty, which is what keeps a bench player's lucky +40 from becoming a rating. Older games
count less: last season half, the season before a quarter.

Walk-forward: ratings for a game use only games before it (refitted every REFIT_DAYS days).

    python -m nbamodel.impact            # tune the penalty, print the top players, test in the game model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CACHE, _get

ROOT = Path(__file__).resolve().parent.parent
REFIT_DAYS = 14
SEASON_WEIGHT = {0: 1.0, 1: 0.5, 2: 0.25}
LAMBDA = 0.3        # ridge penalty, chosen on the tuning seasons (2017-20) from 0.3, 1 and 3


def box(season: int, refresh: bool = False) -> pd.DataFrame:
    p = pd.read_parquet(_get(f"player_box/parquet/player_box_{season}.parquet", CACHE / f"player_box_{season}.parquet", refresh))
    p = p[(p.season_type == 2)].copy()
    p["minutes"] = pd.to_numeric(p.minutes, errors="coerce").fillna(0.0)
    p = p[p.minutes > 0]
    p["pm"] = pd.to_numeric(p.plus_minus.astype(str).str.replace("+", "", regex=False), errors="coerce")
    p = p.dropna(subset=["pm"])
    p["game_id"] = p.game_id.astype(str)
    p["date"] = pd.to_datetime(p.game_date)
    p["home"] = (p.home_away == "home").astype(int)
    p["season"] = season
    return p[["game_id", "season", "date", "team_id", "athlete_id", "athlete_display_name", "minutes", "pm", "home"]]


def _rows(pb: pd.DataFrame):
    """Yield (game rows) arrays per game: for each player-game, the column indices and coefficients."""
    out = []
    for gid, g in pb.groupby("game_id", sort=False):
        teams = g.team_id.unique()
        if len(teams) != 2:
            continue
        tm = {t: g[g.team_id == t] for t in teams}
        tot = {t: float(tm[t].minutes.sum()) for t in teams}
        if min(tot.values()) < 200:
            continue
        for t in teams:
            o = teams[0] if t == teams[1] else teams[1]
            mine, opp = tm[t], tm[o]
            for r in mine.itertuples():
                ids = list(mine.athlete_id) + list(opp.athlete_id)
                own = np.where(mine.athlete_id.to_numpy() == r.athlete_id, 1.0,
                               np.minimum(1.0, 4.0 * mine.minutes.to_numpy() / max(tot[t] - r.minutes, 1.0)))
                ov = -5.0 * opp.minutes.to_numpy() / tot[o]
                coef = np.concatenate([own, ov]) * (r.minutes / 48.0)
                out.append((gid, r.season, r.date, ids, coef, (1 if r.home else -1) * r.minutes / 48.0, float(r.pm), r.minutes))
    return out


def fit(rows, index, season_now, lam=LAMBDA, prior=None):
    """Weighted ridge on the player-game rows. index: athlete -> column. Returns (ratings dict, home)."""
    n = len(index) + 1                      # last column: home court
    A = np.zeros((n, n))
    b = np.zeros(n)
    for (_, season, _, ids, coef, hc, y, m) in rows:
        w = SEASON_WEIGHT.get(season_now - season, 0.0) / max(m, 4.0)
        if w == 0:
            continue
        cols = np.array([index[a] for a in ids] + [n - 1])
        x = np.append(coef, hc)
        A[np.ix_(cols, cols)] += w * np.outer(x, x)
        b[cols] += w * x * y
    pr = np.zeros(n)
    if prior:
        for a, i in index.items():
            pr[i] = prior.get(a, 0.0)
    P = np.full(n, lam)
    P[-1] = 1e-6
    sol = np.linalg.solve(A + np.diag(P), b + P * pr)
    inv = {i: a for a, i in index.items()}
    return {inv[i]: float(sol[i]) for i in range(n - 1)}, float(sol[-1])


def season_rows(season: int, refresh: bool = False):
    """Player-game rows for one season, cached (rebuilt when refresh, for the season in progress)."""
    import pickle
    p = CACHE / "exp" / f"impact_rows_{season}.pkl"
    if p.exists() and not refresh:
        return pickle.loads(p.read_bytes())
    rows = _rows(box(season, refresh=refresh))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(rows))
    return rows


def _accumulate(rows, index, n):
    A = np.zeros((n, n))
    b = np.zeros(n)
    for (_, _, _, ids, coef, hc, y, m) in rows:
        w = 1.0 / max(m, 4.0)
        cols = np.array([index[a] for a in ids] + [n - 1])
        x = np.append(coef, hc)
        A[np.ix_(cols, cols)] += w * np.outer(x, x)
        b[cols] += w * x * y
    return A, b


def walk(seasons, lam=LAMBDA, current=None):
    """Walk-forward ratings. Returns (fits, index): fits is a list of (from_date, ratings array, home)
    in date order; a game on date d uses the last fit with from_date <= d. `current`: the season in
    progress, whose rows are rebuilt from fresh box scores."""
    allrows = {s: season_rows(s, refresh=(s == current)) for s in range(min(seasons) - 2, max(seasons) + 1)}
    ids = sorted({a for rs in allrows.values() for r in rs for a in r[3]})
    index = {a: i for i, a in enumerate(ids)}
    n = len(index) + 1
    full = {}
    fits = []
    P = np.full(n, lam)
    P[-1] = 1e-6
    for s in seasons:
        for back in (1, 2):
            if s - back not in full:
                full[s - back] = _accumulate(allrows.get(s - back, []), index, n)
        A0 = SEASON_WEIGHT[1] * full[s - 1][0] + SEASON_WEIGHT[2] * full[s - 2][0]
        b0 = SEASON_WEIGHT[1] * full[s - 1][1] + SEASON_WEIGHT[2] * full[s - 2][1]
        cur = sorted(allrows.get(s, []), key=lambda r: r[2])
        if not cur:
            continue
        dates = pd.to_datetime([r[2] for r in cur])
        start = dates.min().normalize()
        cuts = list(pd.date_range(start, dates.max() + pd.Timedelta(days=1), freq=f"{REFIT_DAYS}D"))
        Ac, bc = np.zeros((n, n)), np.zeros(n)
        k = 0
        for c in cuts:
            chunk = []
            while k < len(cur) and dates[k] < c:
                chunk.append(cur[k])
                k += 1
            if chunk:
                a_, b_ = _accumulate(chunk, index, n)
                Ac += a_
                bc += b_
            sol = np.linalg.solve(A0 + Ac + np.diag(P), b0 + bc)
            fits.append((c, sol[:-1].copy(), float(sol[-1])))
        full[s] = (Ac, bc)
    return fits, index


def ratings_on(fits, day):
    """The fit in force on `day` (the latest one dated on or before it)."""
    best = None
    for f in fits:
        if f[0] <= day:
            best = f
        else:
            break
    return best


def game_strength(seasons, fits, index, current=None):
    """For every game: each side's lineup strength, 5 x the minutes-weighted average rating of the
    players who played, weighted by their usual minutes (average over the team's last 15 games they
    played, or 12 for a player new to the team). Known before tip-off once the lineup is known,
    the same as the missing-players feature."""
    from collections import defaultdict, deque
    out = []
    for s in seasons:
        pb = box(s, refresh=(s == current)).sort_values(["date", "game_id"])
        usual = defaultdict(lambda: deque(maxlen=15))     # (team, athlete) -> recent minutes
        fi = 0
        cur = None
        for (day, gid), g in pb.groupby(["date", "game_id"], sort=True):
            while fi < len(fits) and fits[fi][0] <= day:
                cur = fits[fi]
                fi += 1
            row = {"game_id": gid}
            for t, tp in g.groupby("team_id"):
                side = "h" if tp.home.iloc[0] == 1 else "a"
                if cur is None:
                    row["imp_" + side] = np.nan
                    continue
                w = np.array([np.mean(usual[(t, a)]) if usual[(t, a)] else 12.0 for a in tp.athlete_id])
                r = np.array([cur[1][index[a]] if a in index else 0.0 for a in tp.athlete_id])
                row["imp_" + side] = float(5 * (w * r).sum() / w.sum())
            out.append(row)
            for r in g.itertuples():
                usual[(r.team_id, r.athlete_id)].append(r.minutes)
    d = pd.DataFrame(out)
    d["imp_diff"] = d.imp_h - d.imp_a
    return d


def history(seasons, lam=LAMBDA):
    """imp_diff for every game of finished seasons, cached by the last season and the penalty."""
    p = CACHE / "exp" / f"impact_hist_{min(seasons)}_{max(seasons)}_{lam:g}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    fits, index = walk(list(seasons), lam)
    d = game_strength(list(seasons), fits, index)[["game_id", "imp_h", "imp_a", "imp_diff"]]
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(p)
    return d


def today(season, out_ids, lam=LAMBDA):
    """Ratings as of today and each team's expected lineup strength tonight: the players who played
    in the team's last 5 games and are not listed out, weighted by their usual minutes.
    Returns (strength by team id, ratings by athlete id, names)."""
    from collections import defaultdict, deque
    fits, index = walk([season], lam, current=season)
    if not fits:
        return {}, {}, {}
    r = fits[-1][1]
    rating = {a: float(r[i]) for a, i in index.items()}
    pb = box(season).sort_values(["date", "game_id"])
    names = dict(zip(pb.athlete_id, pb.athlete_display_name))
    last_team = dict(zip(pb.athlete_id, pb.team_id))
    usual = defaultdict(lambda: deque(maxlen=15))
    for x in pb.itertuples():
        usual[(x.team_id, x.athlete_id)].append(x.minutes)
    strength = {}
    for tid, tp in pb.groupby("team_id"):
        last5 = tp.game_id.drop_duplicates().tail(5)
        ids = [a for a in tp[tp.game_id.isin(last5)].athlete_id.unique() if last_team.get(a) == tid and a not in out_ids]
        if not ids:
            continue
        w = np.array([np.mean(usual[(tid, a)]) for a in ids])
        v = np.array([rating.get(a, 0.0) for a in ids])
        strength[int(tid)] = float(5 * (w * v).sum() / w.sum())
    return strength, rating, names
