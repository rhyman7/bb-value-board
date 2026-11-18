"""Past spreads and totals from the Odds API, one snapshot per game day at 16:00 UTC (11 AM ET in
winter), for the seasons in data/odds/plan.json. A snapshot holds every upcoming game with each US
book's line at that moment, so a game's earliest snapshot is our stand-in for its opener (at most a
day before tip). Cost: 10 credits per market per snapshot, so 20 a day for spreads and totals.

Resumable: days already in the CSV are skipped. Stops before MAX_CREDITS (env, default 13000).

    python -m oddsapi.history                 # writes data/odds/<league>_<season>.csv.gz
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from . import client

DIR = Path(__file__).resolve().parent.parent / "data" / "odds"
COLS = ["snapshot", "event_id", "commence_time", "home_team", "away_team", "book", "last_update",
        "home_spread", "home_spread_price", "away_spread_price", "total", "over_price", "under_price"]


def rows_from(snap, hours=40):
    """One row per game and book; only games tipping within `hours` of the snapshot (later games show
    up again in the next days' snapshots, and this keeps the files small)."""
    out = []
    ts = snap.get("timestamp")
    t0 = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts and ts != "t" else None
    for e in snap.get("data", []):
        if t0 is not None:
            tip = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            if tip < t0 or tip > t0 + timedelta(hours=hours):
                continue
        for b in e.get("bookmakers", []):
            r = {"snapshot": ts, "event_id": e["id"], "commence_time": e["commence_time"], "home_team": e["home_team"],
                 "away_team": e["away_team"], "book": b["key"], "last_update": b.get("last_update")}
            for m in b.get("markets", []):
                if m["key"] == "spreads":
                    for o in m["outcomes"]:
                        if o["name"] == e["home_team"]:
                            r["home_spread"], r["home_spread_price"] = o.get("point"), o.get("price")
                        elif o["name"] == e["away_team"]:
                            r["away_spread_price"] = o.get("price")
                elif m["key"] == "totals":
                    for o in m["outcomes"]:
                        if o["name"].lower() == "over":
                            r["total"], r["over_price"] = o.get("point"), o.get("price")
                        elif o["name"].lower() == "under":
                            r["under_price"] = o.get("price")
            out.append(r)
    return out


def run():
    cap = int(os.environ.get("MAX_CREDITS", "13000"))
    plan = json.loads((DIR / "plan.json").read_text())
    for key, days in plan.items():
        sport = client.SPORTS[key.split("_")[0]]
        path = DIR / f"{key}.csv.gz"
        done = set()
        if path.exists():
            with gzip.open(path, "rt", newline="") as f:
                done = {r["snapshot_day"] for r in csv.DictReader(f)}
        new = not path.exists()
        with gzip.open(path, "at", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["snapshot_day"] + COLS)
            if new:
                w.writeheader()
            for d in days:
                if d in done:
                    continue
                if client.used_total + 20 > cap:
                    print(f"stopping: {client.used_total} credits used, cap {cap}")
                    return
                try:
                    snap, _ = client.get(f"/historical/sports/{sport}/odds", regions="us", markets="spreads,totals",
                                         oddsFormat="american", date=f"{d}T16:00:00Z")
                except Exception as ex:
                    print("  failed", key, d, repr(ex))
                    continue
                rows = rows_from(snap)
                for r in rows:
                    w.writerow({"snapshot_day": d, **r})
                if not rows:
                    w.writerow({"snapshot_day": d})      # marks the day as fetched, so a rerun doesn't pay again
                f.flush()
        print(key, "done; credits used so far", client.used_total)


if __name__ == "__main__":
    run()
    print("total credits used:", client.used_total, "remaining:", client.remaining()[0])
