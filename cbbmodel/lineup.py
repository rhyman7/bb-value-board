"""College lineup strength: how good the players a team is expected to have tonight are, from their
box scores (there is no plus-minus in college box scores, so no impact ratings like the NBA's).

Expected players: those who played in the team's last 5 games, less anyone who has sat the last 2
in a row (the rule the board uses for who is out, with no injury report) and anyone ESPN lists out.
Each is rated by Game Score per 40 minutes, this season to date plus half of last season, shrunk
toward a replacement-level player with 150 minutes, and weighted by his usual minutes (last 10
games). Strength = 5 x the weighted average. The model uses home minus away (lineup_diff).

Tested Nov 18, 2026 (experiments/cbb_lineup.py): holdout 2019-23 home games, margin miss
-0.008 +/- 0.003; tuning 2016-18 -0.007 +/- 0.006.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from . import availability, data

REPL, PRIOR = 0.28, 150.0


def _rate(tot, prev, a):
    g_, m_ = tot.get(a, (0.0, 0.0))
    pg, pm = prev.get(a, (0.0, 0.0))
    return (g_ + 0.5 * pg + REPL * PRIOR) / (m_ + 0.5 * pm + PRIOR) * 40


def _strength(ids, tid, usual, tot, prev):
    w = np.array([np.mean(usual[(tid, a)]) for a in ids])
    if not len(ids) or w.sum() == 0:
        return None
    r = np.array([_rate(tot, prev, a) for a in ids])
    return float(5 * (w * r).sum() / w.sum())


def _season_rows(season, prev, refresh=False):
    """Per game {team: strength} before tip-off, and the season's final totals."""
    pb = availability.box(season, refresh=refresh)
    pb = pb[pb.minutes > 0].sort_values(["date", "game_id"])
    tot = defaultdict(lambda: [0.0, 0.0])
    recent = defaultdict(lambda: deque(maxlen=5))
    usual = defaultdict(lambda: deque(maxlen=10))
    out = {}
    for _, todays in pb.groupby("date"):
        for gid, g in todays.groupby("game_id"):
            vals = {}
            for t in g.team_id.unique():
                games = list(recent[t])
                if not games:
                    continue
                cand = set().union(*games)
                ids = [a for a in cand if not (len(games) >= 2 and a not in games[-1] and a not in games[-2])]
                v = _strength(ids, t, usual, tot, prev)
                if v is not None:
                    vals[int(t)] = v
            out[str(gid)] = vals
        for x in todays.itertuples():
            t = tot[x.athlete_id]
            t[0] += x.gmsc
            t[1] += x.minutes
            usual[(x.team_id, x.athlete_id)].append(x.minutes)
        for (_, t), g in todays.groupby(["game_id", "team_id"]):
            recent[t].append(set(g.athlete_id))
    return out, {a: (v[0], v[1]) for a, v in tot.items()}, (tot, recent, usual)


def history(g, seasons, current=None):
    """lineup_diff for every game of `g` in `seasons` (home minus away; NaN when unknown)."""
    prev = {}
    try:
        _, prev, _ = _season_rows(min(seasons) - 1, {})
    except Exception:
        pass
    st = {}
    for s in seasons:
        try:
            rows, final, _ = _season_rows(s, prev, refresh=(s == current))
        except Exception:
            continue
        st.update(rows)
        prev = final
    out = []
    for x in g[g.season.isin(seasons)].itertuples():
        v = st.get(str(x.game_id), {})
        h, a = v.get(int(x.home_id)), v.get(int(x.away_id))
        out.append({"game_id": str(x.game_id), "lineup_h": h, "lineup_a": a,
                    "lineup_diff": (h - a) if h is not None and a is not None else np.nan})
    return pd.DataFrame(out)


def today(season, out_ids=()):
    """Each team's expected lineup strength for its next game. Returns {team_id: strength}."""
    try:
        _, prev, _ = _season_rows(season - 1, {})
    except Exception:
        prev = {}
    _, _, (tot, recent, usual) = _season_rows(season, prev, refresh=True)
    out_ids = set(out_ids)
    res = {}
    for t, games in recent.items():
        games = list(games)
        cand = set().union(*games) if games else set()
        ids = [a for a in cand if a not in out_ids and not (len(games) >= 2 and a not in games[-1] and a not in games[-2])]
        v = _strength(ids, t, usual, {a: tuple(v) for a, v in tot.items()}, prev)
        if v is not None:
            res[int(t)] = v
    return res
