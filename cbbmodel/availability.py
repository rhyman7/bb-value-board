"""Missing players for college games. There is no college injury report, so on the board a regular
counts as out only when he has sat his team's last INFER_K games in a row (see
experiments/cbb_injuries.py for how often that is right). Values use the NBA method
(nbamodel/availability.py) on college box scores.
"""
from __future__ import annotations

import pandas as pd

from nbamodel import availability as nba_av

from . import data, roster

INFER_K = 2


def box(season, refresh=False):
    """Player box in the shape nbamodel.availability expects."""
    p = pd.read_parquet(data._get(f"player_box/parquet/player_box_{season}.parquet", data.CACHE / f"player_box_{season}.parquet", refresh))
    p = p[p.season_type.isin([2, 3])].copy()
    num = ["minutes", "points", "field_goals_made", "field_goals_attempted", "free_throws_made", "free_throws_attempted",
           "offensive_rebounds", "defensive_rebounds", "steals", "assists", "blocks", "fouls", "turnovers"]
    for c in num:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    p["gmsc"] = (p.points + 0.4 * p.field_goals_made - 0.7 * p.field_goals_attempted
                 - 0.4 * (p.free_throws_attempted - p.free_throws_made) + 0.7 * p.offensive_rebounds
                 + 0.3 * p.defensive_rebounds + p.steals + 0.7 * p.assists + 0.7 * p.blocks
                 - 0.4 * p.fouls - p.turnovers)
    p["team_id"] = pd.to_numeric(p.team_id, errors="coerce")
    p["athlete_id"] = pd.to_numeric(p.athlete_id, errors="coerce")
    p = p.dropna(subset=["team_id", "athlete_id"])
    p["team_id"] = p.team_id.astype(int)
    p["athlete_id"] = p.athlete_id.astype(int)
    p["game_id"] = p.game_id.astype(str)
    p["date"] = pd.to_datetime(p.game_date)
    p["season"] = season
    return p[["game_id", "season", "date", "team_id", "athlete_id", "athlete_display_name", "minutes", "gmsc", "starter"]]


def missing_value(g, seasons, infer_k=INFER_K):
    return nba_av.missing_value(g, seasons, box_fn=box, infer_k=infer_k)


ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"


def _json(url):
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "bb-value-board"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def _collect(blocks, out):
    """ESPN injury blocks: [{team, injuries: [{athlete: {id or links}, status}]}]. Out and Doubtful count."""
    for team in blocks or []:
        for it in team.get("injuries", []) or []:
            st = (it.get("status") or (it.get("type") or {}).get("description") or "").lower()
            aid = nba_av.espn_athlete_id(it.get("athlete"))
            if aid and any(w in st for w in ("out", "doubtful")):
                out[int(aid)] = st


def espn_injuries(game_ids, limit=60):
    """Whatever ESPN lists for college: the league injuries page, then each game's summary (which
    carries an injuries block when ESPN has one). Returns ({athlete_id: status}, note). ESPN's
    college coverage is thin, so this only adds to the inferred list, never replaces it."""
    out, ok = {}, 0
    try:
        _collect(_json(f"{ESPN}/injuries").get("injuries"), out)
        ok += 1
    except Exception:
        pass
    for gid in list(game_ids)[:limit]:
        try:
            _collect(_json(f"{ESPN}/summary?event={gid}").get("injuries"), out)
            ok += 1
        except Exception:
            continue
    return out, f"ESPN injury data read for {ok} of {min(len(game_ids), limit) + 1} requests, {len(out)} players listed Out/Doubtful"


def current(season, extra_out=(), refresh=True):
    """Who is out for each team today and what they are worth, valued like the backtest.
    Out = a regular who has sat the team's last INFER_K games in a row, or one ESPN lists Out or
    Doubtful (extra_out). Returns (miss {team: value}, detail {team: [(name, value)]}, out_ids)."""
    import numpy as np
    pb = box(season, refresh=refresh)
    played = pb[pb.minutes > 0].sort_values(["date", "game_id"])
    try:
        prev = box(season - 1)
        prev_tot = prev[prev.minutes > 0].groupby("athlete_id")[["gmsc", "minutes"]].sum()
    except Exception:
        prev_tot = pd.DataFrame(columns=["gmsc", "minutes"])
    tot = played.groupby("athlete_id")[["gmsc", "minutes"]].sum()
    last_team = played.groupby("athlete_id").team_id.last()
    names = played.groupby("athlete_id").athlete_display_name.last()
    extra_out = set(extra_out)
    miss, detail, out_ids = {}, {}, set()
    for tid, tp in played.groupby("team_id"):
        recent = list(tp.game_id.drop_duplicates())[-nba_av.WINDOW:]
        games = [set(tp.loc[tp.game_id == gid, "athlete_id"]) for gid in recent]
        r = tp[tp.game_id.isin(recent)]
        for aid, pr in r.groupby("athlete_id"):
            if last_team.get(aid) != tid:
                continue
            share = len(pr) / max(len(games), 1)
            role = pr.minutes.mean()
            if share < 0.3 or role < 10:
                continue
            streak = 0
            for gm in reversed(games):
                if aid in gm:
                    break
                streak += 1
            if streak < INFER_K and aid not in extra_out:
                continue
            out_ids.add(int(aid))
            g_, m_ = tot.loc[aid] if aid in tot.index else (0.0, 0.0)
            pg, pm = prev_tot.loc[aid] if aid in prev_tot.index else (0.0, 0.0)
            rate = (g_ + 0.5 * pg + nba_av.REPLACEMENT * nba_av.PRIOR_MIN) / (m_ + 0.5 * pm + nba_av.PRIOR_MIN)
            v = float(max(0.0, rate - nba_av.REPLACEMENT) * role)
            miss[int(tid)] = miss.get(int(tid), 0.0) + v
            if v > 0.3:
                why = "out on ESPN" if aid in extra_out else f"missed last {streak}"
                detail.setdefault(int(tid), []).append((f"{names.get(aid, aid)} ({why})", v))
    return miss, detail, out_ids
