"""Player impact from real lineups: rebuild who was on the court from the play-by-play
substitutions, cut each game into stints (stretches with the same ten players), and give
impact.py one row per stint instead of one per player-game:

    home points - away points in the stint = (minutes / 48) * (sum of the home five's ratings
                                              - sum of the away five's + home court)

The play-by-play names who enters and leaves, but not who starts a quarter after the first. A
quarter's starters are the players who show up in it (in any play, or leaving the game) before they
enter it, topped up from whoever finished the quarter before.

    python -m nbamodel.pbp_impact 2025      # build and cache one season's stints, print a summary
"""
from __future__ import annotations

import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from .data import CACHE, _get
from .impact import box as player_box

COLS = ["game_id", "game_play_number", "type_text", "period_number", "clock_minutes", "clock_seconds", "home_score",
        "away_score", "athlete_id_1", "athlete_id_2", "athlete_id_3", "season_type", "home_team_id", "away_team_id"]


def pbp(season: int, refresh: bool = False) -> pd.DataFrame:
    p = _get(f"pbp/parquet/play_by_play_{season}.parquet", CACHE / f"pbp_{season}.parquet", refresh)
    d = pd.read_parquet(p, columns=COLS)
    d = d[d.season_type == 2].copy()
    d["game_id"] = d.game_id.astype(str)
    return d


def _elapsed(period, mins, secs):
    left = mins * 60 + secs
    return np.where(period <= 4, (period - 1) * 720 + (720 - left), 2880 + (period - 5) * 300 + (300 - left))


def game_stints(ev: pd.DataFrame, team_of: dict, starters: dict, home: int):
    """ev: one game's plays in order. team_of: athlete -> team; starters: team -> set of 5.
    Returns [(home five, away five, minutes, home pts - away pts)]."""
    away = next((t for t in starters if t != home), None)
    if away is None or any(len(s) != 5 for s in starters.values()):
        return []
    ev = ev.sort_values("game_play_number")
    t = _elapsed(ev.period_number.to_numpy(), ev.clock_minutes.to_numpy(float), ev.clock_seconds.to_numpy(float))
    hs, as_ = ev.home_score.ffill().fillna(0).to_numpy(float), ev.away_score.ffill().fillna(0).to_numpy(float)
    sub = ev.type_text.str.contains("Substitution", na=False).to_numpy()
    a1, a2, a3 = (ev[c].to_numpy() for c in ("athlete_id_1", "athlete_id_2", "athlete_id_3"))
    per = ev.period_number.to_numpy()
    out = []
    on = {home: list(starters[home]), away: list(starters[away])}
    for p in sorted(set(per)):
        idx = np.where(per == p)[0]
        if p > 1:       # who starts this quarter
            seen, entered = {home: [], away: []}, set()
            for i in idx:
                if sub[i]:
                    for a, enters in ((a1[i], True), (a2[i], False)):
                        if pd.isna(a):
                            continue
                        a = int(a)
                        tm = team_of.get(a)
                        if tm not in seen:
                            continue
                        if enters:
                            entered.add(a)
                        elif a not in entered and a not in seen[tm]:
                            seen[tm].append(a)
                else:
                    for a in (a1[i], a2[i], a3[i]):
                        if pd.isna(a):
                            continue
                        a = int(a)
                        tm = team_of.get(a)
                        if tm in seen and a not in entered and a not in seen[tm]:
                            seen[tm].append(a)
            for tm in (home, away):
                five = seen[tm][:5]
                for a in on[tm]:
                    if len(five) >= 5:
                        break
                    if a not in five and a not in entered:
                        five.append(a)
                if len(five) < 5:
                    return out          # can't place five players: stop this game here
                on[tm] = five
        start_t = (p - 1) * 720 if p <= 4 else 2880 + (p - 5) * 300
        end_t = start_t + (720 if p <= 4 else 300)
        i0 = idx[0]
        s_t, s_h, s_a = start_t, hs[i0 - 1] if i0 > 0 else 0.0, as_[i0 - 1] if i0 > 0 else 0.0
        for i in idx:
            if not sub[i] or pd.isna(a1[i]) or pd.isna(a2[i]):
                continue
            inn, outp = int(a1[i]), int(a2[i])
            tm = team_of.get(inn)
            if tm not in on or outp not in on[tm]:
                continue
            # score before this substitution: the last play before it in order
            h_, a_ = (hs[i - 1], as_[i - 1]) if i > 0 else (0.0, 0.0)
            if t[i] > s_t:
                out.append((tuple(on[home]), tuple(on[away]), (t[i] - s_t) / 60.0, (h_ - s_h) - (a_ - s_a)))
                s_t, s_h, s_a = t[i], h_, a_
            on[tm] = [inn if x == outp else x for x in on[tm]]
        h_, a_ = hs[idx[-1]], as_[idx[-1]]
        if end_t > s_t:
            out.append((tuple(on[home]), tuple(on[away]), (end_t - s_t) / 60.0, (h_ - s_h) - (a_ - s_a)))
    return out


def season_stints(season: int, refresh: bool = False):
    """Rows in impact.py's format: (game_id, season, date, ids, coef, home_coef, y, minutes)."""
    path = CACHE / "exp" / f"pbp_stints_{season}.pkl"
    if path.exists() and not refresh:
        return pickle.loads(path.read_bytes())
    ev = pbp(season, refresh)
    raw = pd.read_parquet(CACHE / f"player_box_{season}.parquet") if (CACHE / f"player_box_{season}.parquet").exists() else None
    pb = player_box(season, refresh)
    starters = {}
    if raw is not None and "starter" in raw:
        r = raw[(raw.season_type == 2) & raw.starter.astype(bool)]
        for (g, tm), x in r.groupby([r.game_id.astype(str), "team_id"]):
            starters.setdefault(g, {})[int(tm)] = set(int(a) for a in x.athlete_id)
    rows, games, ok = [], 0, 0
    for gid, g in pb.groupby("game_id"):
        e = ev[ev.game_id == gid] if False else None
    ev_by = {k: v for k, v in ev.groupby("game_id")}
    for gid, g in pb.groupby("game_id"):
        games += 1
        e = ev_by.get(gid)
        st = starters.get(gid)
        if e is None or not st:
            continue
        home = int(e.home_team_id.iloc[0])
        team_of = {int(a): int(t) for a, t in zip(g.athlete_id, g.team_id)}
        date, ssn = g.date.iloc[0], season
        stints = game_stints(e, team_of, {int(k): v for k, v in st.items()}, home)
        if stints:
            ok += 1
        for h5, a5, mins, y in stints:
            if mins <= 0:
                continue
            coef = np.array([1.0] * 5 + [-1.0] * 5) * mins / 48.0
            rows.append((gid, ssn, date, list(h5) + list(a5), coef, mins / 48.0, float(y), mins))
    print(f"{season}: {ok} of {games} games rebuilt, {len(rows)} stints")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(rows))
    return rows


if __name__ == "__main__":
    rows = season_stints(int(sys.argv[1]), refresh=True)
    m = np.array([r[7] for r in rows])
    print("minutes per game", round(m.sum() / len({r[0] for r in rows}), 1), "median stint", round(float(np.median(m)), 2))
