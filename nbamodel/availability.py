"""Who is missing: the value of a team's regular players who don't play in a game.

For each team and game, the pool is the players who played for the team in its previous 15 games
and have not played for another team since. A player's value is his minutes when he plays (over
those games) times his production per minute above a replacement level, from Hollinger's Game Score
over his season so far (last season's numbers stand in for the first 10 games, shrunk toward
replacement). The team's missing value is the sum over pool players who do not play.

In the backtest the model knows who played, the same way the football backtest knows the starting
quarterback. On the board it comes from the injury report.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .data import CACHE, _get

WINDOW = 15
REPLACEMENT = 0.28      # Game Score per minute of a replacement-level player (about the 20th percentile of regulars)
PRIOR_MIN = 300.0       # minutes of replacement-level production a player's rate is shrunk with


def player_box(season: int, refresh: bool = False) -> pd.DataFrame:
    p = pd.read_parquet(_get(f"player_box/parquet/player_box_{season}.parquet", CACHE / f"player_box_{season}.parquet", refresh))
    p = p[p.season_type == 2].copy()
    num = ["minutes", "points", "field_goals_made", "field_goals_attempted", "free_throws_made", "free_throws_attempted",
           "offensive_rebounds", "defensive_rebounds", "steals", "assists", "blocks", "fouls", "turnovers"]
    for c in num:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    p["gmsc"] = (p.points + 0.4 * p.field_goals_made - 0.7 * p.field_goals_attempted
                 - 0.4 * (p.free_throws_attempted - p.free_throws_made) + 0.7 * p.offensive_rebounds
                 + 0.3 * p.defensive_rebounds + p.steals + 0.7 * p.assists + 0.7 * p.blocks
                 - 0.4 * p.fouls - p.turnovers)
    p["game_id"] = p.game_id.astype(str)
    p["date"] = pd.to_datetime(p.game_date)
    p["season"] = season
    return p[["game_id", "season", "date", "team_id", "athlete_id", "athlete_display_name", "minutes", "gmsc", "starter"]]


def missing_value(g: pd.DataFrame, seasons, box_fn=None, infer_k=None) -> pd.DataFrame:
    """Missing value for home and away in each game of `g` (columns miss_h, miss_a, plus the names).
    box_fn: the player box loader (NBA by default). infer_k: when set, a regular counts as missing
    only if he has sat the team's last infer_k games in a row (what can be known with no injury
    report), instead of whether he actually played."""
    rows = []
    prev_rate = {}   # athlete -> (gmsc, minutes) last season
    box_fn = box_fn or player_box
    for season in seasons:
        try:
            pb = box_fn(season, refresh=False)
        except Exception:
            continue
        played = pb[pb.minutes > 0].sort_values(["date", "game_id"])
        sg = g[g.season == season].sort_values(["date", "game_id"])
        # per-player running totals this season, last team, and each team's recent games
        tot = defaultdict(lambda: [0.0, 0.0])            # athlete -> [gmsc, minutes]
        last_team = {}
        recent = defaultdict(lambda: deque(maxlen=WINDOW))  # team -> deque of {athlete: minutes}
        names = {}
        by_game = {gid: grp for gid, grp in played.groupby("game_id")}
        for day, todays in sg.groupby("date"):
            day_rows = []
            for x in todays.itertuples():
                gp = by_game.get(x.game_id)
                out = {"game_id": x.game_id}
                for side, tid in (("h", x.home_id), ("a", x.away_id)):
                    here = set(gp.loc[gp.team_id == tid, "athlete_id"]) if gp is not None else set()
                    games = recent[tid]
                    mins = defaultdict(list)
                    for gm in games:
                        for aid, m in gm.items():
                            mins[aid].append(m)
                    miss, who = 0.0, []
                    if infer_k:
                        streak = {}
                        for aid in mins:
                            n = 0
                            for gm in reversed(games):
                                if aid in gm:
                                    break
                                n += 1
                            streak[aid] = n
                    for aid, ms in mins.items():
                        if last_team.get(aid) != tid:
                            continue
                        if (streak[aid] < infer_k) if infer_k else (aid in here):
                            continue
                        role = float(np.mean(ms))  # minutes when he plays
                        share = len(ms) / max(len(games), 1)
                        if share < 0.3 or role < 10:
                            continue
                        g_, m_ = tot[aid]
                        pg, pm = prev_rate.get(aid, (0.0, 0.0))
                        # shrink toward replacement with PRIOR_MIN minutes; last season counts half
                        rate = (g_ + 0.5 * pg + REPLACEMENT * PRIOR_MIN) / (m_ + 0.5 * pm + PRIOR_MIN)
                        v = max(0.0, rate - REPLACEMENT) * role
                        miss += v
                        if v > 2:
                            who.append(names.get(aid, str(aid)))
                    out["miss_" + side] = miss
                    out["out_" + side] = ", ".join(who[:4])
                day_rows.append(out)
            rows.extend(day_rows)
            # after the day: add its games to the running state
            for x in todays.itertuples():
                gp = by_game.get(x.game_id)
                if gp is None:
                    continue
                for tid in (x.home_id, x.away_id):
                    tp = gp[gp.team_id == tid]
                    recent[tid].append(dict(zip(tp.athlete_id, tp.minutes)))
                    for r in tp.itertuples():
                        t = tot[r.athlete_id]
                        t[0] += r.gmsc
                        t[1] += r.minutes
                        last_team[r.athlete_id] = tid
                        names[r.athlete_id] = r.athlete_display_name
        prev_rate = {aid: (v[0], v[1]) for aid, v in tot.items()}
    return pd.DataFrame(rows)
