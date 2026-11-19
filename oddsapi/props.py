"""Live NBA player props: points, rebounds, assists and threes for today's games, from every US book
the Odds API carries, reduced to one consensus line per player and market (the median line, with
the median over and under prices at it).

Since 2026-11-18 the same call also asks for first-half spreads and totals (Ryan chose to log them
live rather than buy history), reduced the same way to one consensus line per game.

Cost: about 6 credits per game (one per market returned). The pipeline calls it once a day, on the
late-afternoon run, so a full slate costs roughly 40 to 90 credits.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timezone

from . import client

MARKETS = {"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists", "player_threes": "threes"}
GAME_MARKETS = ["spreads_h1", "totals_h1"]
GAME_LINES = {}     # filled by today_props: {game_id: {"h1_spread", "h1_spread_home_price", ...}}


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
                               markets=",".join(list(MARKETS) + GAME_MARKETS), oddsFormat="american")
        except Exception as ex:
            print("  props failed for", home, away, repr(ex))
            continue
        GAME_LINES[gid] = _game_lines(ev, home)
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
            g.setdefault(who, {})[stat] = {"line": line, "over": median_price(q[1] for q in at),
                                           "under": median_price(q[2] for q in at), "books": len(qs)}
        out[gid] = g
    return out


def median_price(prices):
    """Median of American prices, taken in decimal odds. A plain median of an even count can land
    between -100 and +100 (median of -110 and +105 is -2.5), which isn't a price."""
    dec = [1 + (p / 100 if p > 0 else 100 / -p) for p in prices if p]
    if not dec:
        return None
    d = statistics.median(dec) - 1
    return int(round(100 * d)) if d >= 1 else int(round(-100 / d))


def _consensus(qs):
    """qs: (line, price_a, price_b) per book -> median line and the median prices at it."""
    line = statistics.median(q[0] for q in qs)
    at = [q for q in qs if q[0] == line] or qs
    return line, median_price(q[1] for q in at), median_price(q[2] for q in at), len(qs)


def _game_lines(ev, home):
    """First-half spread (+ = home favored, our sign) and total, consensus across books."""
    sp, tot = [], []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            o = {x["name"]: x for x in m.get("outcomes", []) if x.get("point") is not None}
            if m["key"] == "spreads_h1" and home in o and len(o) == 2:
                away = next(x for k, x in o.items() if k != home)
                sp.append((-float(o[home]["point"]), int(o[home]["price"]), int(away["price"])))
            elif m["key"] == "totals_h1" and "Over" in o and "Under" in o and o["Over"]["point"] == o["Under"]["point"]:
                tot.append((float(o["Over"]["point"]), int(o["Over"]["price"]), int(o["Under"]["price"])))
    out = {}
    if sp:
        out["h1_spread"], out["h1_home_price"], out["h1_away_price"], out["h1_spread_books"] = _consensus(sp)
    if tot:
        out["h1_total"], out["h1_over_price"], out["h1_under_price"], out["h1_total_books"] = _consensus(tot)
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
