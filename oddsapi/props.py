"""Live NBA player props: points, rebounds, assists and threes for today's games, from every US book
the Odds API carries, reduced to one consensus line per player and market (the median line, with
the median over and under prices at it).

Cost: about 4 credits per game (one per market returned). The pipeline calls it once a day, on the
late-afternoon run, so a full slate costs roughly 30 to 60 credits.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone

from . import client

MARKETS = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists", "player_threes": "threes"}


def _norm(name):
    return "".join(c for c in name.lower().replace("jr.", "").replace("sr.", "").replace("iii", "").replace("ii", "") if c.isalnum())


def today_props(game_teams, day_iso):
    """game_teams: {game_id: (home_full_name, away_full_name)} for today's games. Returns
    {game_id: {normalized player name: {stat: {"line", "over", "under", "books"}}}}."""
    events, _ = client.get(f"/sports/{client.SPORTS['nba']}/events")
    by_teams = {}
    for e in events:
        d = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(timezone.utc)
        by_teams[(e["home_team"], e["away_team"])] = e["id"]
    out = {}
    for gid, (home, away) in game_teams.items():
        eid = by_teams.get((home, away))
        if not eid:
            continue
        try:
            ev, _ = client.get(f"/sports/{client.SPORTS['nba']}/events/{eid}/odds", regions="us",
                               markets=",".join(MARKETS), oddsFormat="american")
        except Exception as ex:
            print("  props failed for", home, away, repr(ex))
            continue
        quotes = {}
        for bk in ev.get("bookmakers", []):
            for m in bk.get("markets", []):
                stat = MARKETS.get(m["key"])
                if not stat:
                    continue
                sides = {}
                for o in m.get("outcomes", []):
                    who = o.get("description") or ""
                    if o.get("point") is None:
                        continue
                    sides.setdefault(who, {})[o["name"].lower()] = (float(o["point"]), int(o["price"]))
                for who, s in sides.items():
                    if "over" in s and "under" in s and s["over"][0] == s["under"][0]:
                        quotes.setdefault((_norm(who), stat), []).append((s["over"][0], s["over"][1], s["under"][1]))
        g = {}
        for (who, stat), qs in quotes.items():
            line = statistics.median(q[0] for q in qs)
            at = [q for q in qs if q[0] == line] or qs
            g.setdefault(who, {})[stat] = {"line": line, "over": int(statistics.median(q[1] for q in at)),
                                           "under": int(statistics.median(q[2] for q in at)), "books": len(qs)}
        out[gid] = g
    return out


def attach(players_by_game, props):
    """Put each player's consensus lines on his projection row (row['props'][stat])."""
    n = 0
    for gid, e in players_by_game.items():
        g = props.get(gid) or {}
        for side in ("home", "away"):
            for r in e.get(side, []):
                p = g.get(_norm(r["name"]))
                if p:
                    r["props"] = p
                    n += 1
    return n
