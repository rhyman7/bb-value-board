"""Sharp-price test: bet a US-regulated book's NBA spread when it is off Pinnacle's no-vig price.

Pinnacle (data/odds/nba_pinnacle.csv.gz) and the US books (nba_2025/2026.csv.gz) were read at the same
11 AM ET snapshot, 2024-25 and 2025-26. Pinnacle's two prices give a no-vig cover chance at its line;
that sets the market's expected margin m (normal, sd 12.5 points), which prices any other line:
P(home covers at spread s) = Phi((m + s) / 12.5). A bet is the side at the best regulated book with
the biggest expected value; it needs EV >= threshold. Graded on the final score and against the
closing line (hoopR's consensus close): did the bet beat where the line closed?

    python -m experiments.sharp_price        # writes docs/sharp_price_test.json
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from experiments.opening_lines import NBA_ALIAS, match, odds
from experiments.props_backtest import decimal
from experiments.props_value import OFFSHORE

ODDS = Path(__file__).resolve().parent.parent / "data" / "odds"
OUT = Path(__file__).resolve().parent.parent / "docs" / "sharp_price_test.json"
SD = 12.5
N = NormalDist()


def read(name):
    with gzip.open(ODDS / name, "rt") as f:
        d = pd.read_csv(f)
    d = d.dropna(subset=["event_id", "home_spread", "home_spread_price", "away_spread_price"])
    d["snap"] = pd.to_datetime(d.snapshot, utc=True)
    return d


def summ(r):
    n = len(r)
    return {"n": int(n), "win": round(float((r > 0).mean()), 4), "roi": round(float(r.mean()), 4),
            "roi_se": round(float(r.std() / np.sqrt(n)), 4)} if n else {"n": 0}


def main():
    from nbamodel import backtest, data
    pin = read("nba_pinnacle.csv.gz")
    pin = pin[pin.book == "pinnacle"].sort_values("snap").groupby("event_id").last().reset_index()
    us = pd.concat([read("nba_2025.csv.gz"), read("nba_2026.csv.gz")])
    us = us[us.snapshot_day.isin(set(pin.snapshot_day)) & ~us.book.isin(OFFSHORE | {"betus", "lowvig"})]
    o = pd.concat([odds(f"nba_{s}").assign(season=s) for s in (2025, 2026)], ignore_index=True)
    sched = pd.concat([data.schedule(s) for s in (2025, 2026)], ignore_index=True)
    ids = match(o, sched, NBA_ALIAS).dropna(subset=["game_id"])[["event_id", "game_id", "season"]]
    bt = backtest.run(2026, verbose=False, use_availability=False)["games"]
    bt["game_id"] = bt.game_id.astype(str)
    res_by = bt.set_index("game_id")[["margin", "close_spread"]]

    rows = []
    for p in pin.itertuples():
        g = us[(us.event_id == p.event_id) & (us.snapshot_day == p.snapshot_day)]
        idrow = ids[ids.event_id == p.event_id]
        if g.empty or idrow.empty or idrow.game_id.iloc[0] not in res_by.index:
            continue
        gid = idrow.game_id.iloc[0]
        margin, close = res_by.loc[gid, ["margin", "close_spread"]]   # margin home-away; close: + = home favored
        a, b = 1 / decimal(p.home_spread_price), 1 / decimal(p.away_spread_price)
        ph = float(a / (a + b))
        ph = min(max(ph, 1e-4), 1 - 1e-4)
        m = SD * N.inv_cdf(ph) - p.home_spread           # market's expected home margin
        best = None
        for q in g.itertuples():
            s = q.home_spread
            for side, dec, pc in (("home", float(decimal(q.home_spread_price)), N.cdf((m + s) / SD)),
                                  ("away", float(decimal(q.away_spread_price)), 1 - N.cdf((m + s) / SD))):
                ev = pc * dec - 1
                if best is None or ev > best["ev"]:
                    best = {"ev": ev, "side": side, "spread": s, "dec": dec, "book": q.book, "pin_line": p.home_spread}
        cover = margin + best["spread"]
        res = cover if best["side"] == "home" else -cover
        ret = 0.0 if res == 0 else (best["dec"] - 1 if res > 0 else -1.0)
        # closing line value: points better than the close, for the side taken (close in our sign: + = home favored)
        clv = None
        if pd.notna(close):
            home_close_spread = -close
            clv = (best["spread"] - home_close_spread) if best["side"] == "home" else (home_close_spread - best["spread"])
        rows.append({**best, "ret": ret, "clv": clv, "season": int(idrow.season.iloc[0]),
                     "line_diff": abs(best["spread"] - p.home_spread)})
    d = pd.DataFrame(rows)
    out = {"games": int(len(d)), "sd": SD}
    for t in (0.0, 0.01, 0.02, 0.03, 0.05):
        x = d[d.ev >= t]
        c = x.dropna(subset=["clv"])
        out[f"ev {int(t * 100)}%+"] = {**summ(x.ret), "avg_clv_pts": round(float(c.clv.mean()), 3) if len(c) else None,
                                       "beat_close_pct": round(100 * float((c.clv > 0).mean()), 1) if len(c) else None,
                                       "by_season": {int(k): summ(v.ret) for k, v in x.groupby("season")},
                                       "off_pinnacle_line": round(100 * float((x.line_diff > 0).mean()), 1) if len(x) else None}
    out["books"] = d[d.ev >= 0.02].book.value_counts().to_dict()
    print(json.dumps(out, indent=1))
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
