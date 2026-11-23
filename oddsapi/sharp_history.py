"""Pinnacle's past NBA spreads, for the sharp-price test: is a US book's line worth betting when it
is off Pinnacle's no-vig price? Same 16:00 UTC (11 AM ET) snapshot days as the US-book pull
(data/odds/nba_2025 and nba_2026), so the two can be matched game by game.

Cost: 10 credits a day (spreads, one bookmaker). Days go in a fixed random order so a pull cut short
by MAX_CREDITS still covers both seasons. Resumable. Approved by Ryan on 2026-11-23 ("you decide
what I should spend 3100 on"), capped at 3,000.

    MAX_CREDITS=3000 python -m oddsapi.sharp_history     # writes data/odds/nba_pinnacle.csv.gz
"""
from __future__ import annotations

import csv
import gzip
import os
import random

from . import client
from .history import COLS, DIR, rows_from

PATH = DIR / "nba_pinnacle.csv.gz"


def days():
    out = []
    for k in ("nba_2025", "nba_2026"):
        with gzip.open(DIR / f"{k}.csv.gz", "rt") as f:
            out += sorted({r["snapshot_day"] for r in csv.DictReader(f)})
    random.Random(11).shuffle(out)
    return out


def run():
    cap = int(os.environ["MAX_CREDITS"])
    done = set()
    if PATH.exists():
        with gzip.open(PATH, "rt", newline="") as f:
            done = {r["snapshot_day"] for r in csv.DictReader(f)}
    new = not PATH.exists()
    tried = found = 0
    with gzip.open(PATH, "at", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snapshot_day"] + COLS)
        if new:
            w.writeheader()
        for d in days():
            if d in done:
                continue
            if client.used_total + 10 > cap:
                print(f"stopping: {client.used_total} credits used, cap {cap}")
                break
            try:
                snap, _ = client.get(f"/historical/sports/{client.SPORTS['nba']}/odds", bookmakers="pinnacle",
                                     markets="spreads", oddsFormat="american", date=f"{d}T16:00:00Z")
            except Exception as ex:
                print("  failed", d, repr(ex))
                continue
            rows = rows_from(snap)
            tried += 1
            found += bool(rows)
            for r in rows or [{}]:
                w.writerow({"snapshot_day": d, **r})
            f.flush()
            if tried == 5 and found == 0:
                print("no Pinnacle lines in the first 5 days; stopping so no more credits are spent")
                break
    print(f"days with Pinnacle lines: {found} of {tried}; credits used {client.used_total}")


if __name__ == "__main__":
    run()
    print("remaining:", client.remaining()[0])
