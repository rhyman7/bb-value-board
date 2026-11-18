"""Past NBA player props from the Odds API, for the props backtest (experiments/props_backtest.py).

Takes the game days in data/odds/props_plan.json (spread evenly over 2023-24 to 2025-26; the Odds
API keeps player props from May 2023). For each day: one historical events call (1 credit) to list
that night's games, then one historical event-odds call per game, taken 60 minutes before tip, for
the markets in PROP_MARKETS. Cost: 10 credits per market per game, so 10 a game for points only and
40 for all four stats.

Resumable: games already in the CSV are skipped. Stops before MAX_CREDITS (env).

    PROP_MARKETS=player_points MAX_CREDITS=1600 python -m oddsapi.prop_history
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import client
from .props import MARKETS

DIR = Path(__file__).resolve().parent.parent / "data" / "odds"
PATH = DIR / "nba_props.csv.gz"
COLS = ["day", "event_id", "commence_time", "snapshot", "home_team", "away_team", "book", "market", "player",
        "line", "over", "under"]
BEFORE = timedelta(minutes=60)


def _iso(t):
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rows_from(day, ev, snap_ts):
    out = []
    for bk in ev.get("bookmakers", []):
        for m in bk.get("markets", []):
            sides = {}
            for o in m.get("outcomes", []):
                if o.get("point") is None:
                    continue
                sides.setdefault(o.get("description") or "", {})[o["name"].lower()] = (float(o["point"]), int(o["price"]))
            for who, s in sides.items():
                if "over" in s and "under" in s and s["over"][0] == s["under"][0]:
                    out.append({"day": day, "event_id": ev["id"], "commence_time": ev["commence_time"], "snapshot": snap_ts,
                                "home_team": ev["home_team"], "away_team": ev["away_team"], "book": bk["key"],
                                "market": m["key"], "player": who, "line": s["over"][0], "over": s["over"][1],
                                "under": s["under"][1]})
    return out


def run():
    cap = int(os.environ["MAX_CREDITS"])
    markets = os.environ.get("PROP_MARKETS", "player_points")
    assert all(m in MARKETS for m in markets.split(","))
    per_game = 10 * len(markets.split(","))
    days = json.loads((DIR / "props_plan.json").read_text())["days"]
    done = set()
    if PATH.exists():
        with gzip.open(PATH, "rt", newline="") as f:
            done = {r["event_id"] for r in csv.DictReader(f)}
    new = not PATH.exists()
    with gzip.open(PATH, "at", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for day in days:
            if client.used_total + 1 + per_game > cap:
                break
            d0 = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
            try:
                evs, _ = client.get(f"/historical/sports/{client.SPORTS['nba']}/events", date=_iso(d0 + timedelta(hours=15)))
            except Exception as ex:
                print("  events failed", day, repr(ex))
                continue
            for e in evs.get("data", []):
                tip = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
                if not (d0 + timedelta(hours=16) <= tip <= d0 + timedelta(hours=36)) or e["id"] in done:
                    continue
                if client.used_total + per_game > cap:
                    print(f"stopping: {client.used_total} credits used, cap {cap}")
                    return
                try:
                    ev, _ = client.get(f"/historical/sports/{client.SPORTS['nba']}/events/{e['id']}/odds", regions="us",
                                       markets=markets, oddsFormat="american", date=_iso(tip - BEFORE))
                except Exception as ex:
                    print("  odds failed", day, e["home_team"], repr(ex))
                    continue
                rows = rows_from(day, ev.get("data", {}), ev.get("timestamp"))
                for r in rows or [{"day": day, "event_id": e["id"], "commence_time": e["commence_time"]}]:
                    w.writerow(r)
                done.add(e["id"])
                f.flush()
    print("days done; credits used", client.used_total)


if __name__ == "__main__":
    run()
    print("total credits used:", client.used_total, "remaining:", client.remaining()[0])
