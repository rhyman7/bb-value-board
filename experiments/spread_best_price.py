"""NBA spreads at the best available number: does shopping the noon line across books turn the
model's side into a winner? Same noon-ET lines and no-injury-hindsight model as the opening-line
test (experiments/opening_lines.py), 2024-26. For the model's side, take the book with the best
line (then the best price at it), US-regulated books only.

    python -m experiments.spread_best_price      # writes docs/spread_best_price.json
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.opening_lines import NBA_ALIAS, ODDS, match, odds
from experiments.props_backtest import decimal
from experiments.props_value import OFFSHORE

OUT = Path(__file__).resolve().parent.parent / "docs" / "spread_best_price.json"


def books(key):
    with gzip.open(ODDS / f"{key}.csv.gz", "rt") as f:
        d = pd.read_csv(f)
    d = d.dropna(subset=["event_id", "home_spread", "home_spread_price", "away_spread_price"])
    d["snap"] = pd.to_datetime(d.snapshot, utc=True)
    last = d.groupby("event_id").snap.transform("max")
    return d[(d.snap == last) & ~d.book.isin(OFFSHORE | {"betus", "lowvig"})]


def summ(r):
    n = len(r)
    return {"n": int(n), "win": round(float((r > 0).mean()), 4), "roi": round(float(r.mean()), 4),
            "roi_se": round(float(r.std() / np.sqrt(n)), 4)} if n else {"n": 0}


def main():
    from nbamodel import backtest, data
    o = pd.concat([odds(f"nba_{s}").assign(season=s) for s in (2025, 2026)], ignore_index=True)
    sched = pd.concat([data.schedule(s) for s in (2025, 2026)], ignore_index=True)
    m = match(o, sched, NBA_ALIAS).dropna(subset=["game_id"])
    bt = backtest.run(2026, verbose=False, use_availability=False)["games"]
    bt["game_id"] = bt.game_id.astype(str)
    x = m.merge(bt[["game_id", "margin", "m_margin"]], on="game_id")
    b = pd.concat([books(f"nba_{s}") for s in (2025, 2026)])
    rows = []
    for r in x.itertuples():
        g = b[b.event_id == r.event_id]
        if g.empty:
            continue
        home = r.m_margin > r.morning_margin                    # model likes the home side at the noon line
        if home:     # home spread s: home covers if margin + s > 0; best = largest s, then best price
            s = g.home_spread.max(); gg = g[g.home_spread == s]; dec = decimal(gg.home_spread_price).max()
            res = r.margin + s
        else:
            s = g.home_spread.min(); gg = g[g.home_spread == s]; dec = decimal(gg.away_spread_price).max()
            res = -(r.margin + s)
        med = -r.morning_margin
        med_res = (r.margin + med) if home else -(r.margin + med)
        rows.append({"best": 0.0 if res == 0 else (dec - 1 if res > 0 else -1.0),
                     "median_at_-110": 0.0 if med_res == 0 else (100 / 110 if med_res > 0 else -1.0),
                     "gap": abs(r.m_margin - r.morning_margin), "season": r.season,
                     "line_gain": (s - med) if home else (med - s)})
    d = pd.DataFrame(rows)
    res = {"games": int(len(d)), "avg_points_gained_by_shopping": round(float(d.line_gain.mean()), 2),
           "median_line_at_-110": summ(d["median_at_-110"]), "best_line_and_price": summ(d.best),
           "best_by_gap": {f"{k}+": summ(d.best[d.gap >= k]) for k in (1, 2, 3, 4)},
           "best_by_season": {int(k): summ(g.best) for k, g in d.groupby("season")}}
    print(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
