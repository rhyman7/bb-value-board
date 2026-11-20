"""Tonight's prop picks: bets where the chance of winning beats the best US-regulated price.

The chance blends the model with the market (docs/nba/prop_dist.json): the model's own chance,
from its projection (nbamodel/propprob.py), gets weight w and the market's no-vig chance (from the
consensus prices) gets 1 - w. The model alone is overconfident, and w = 0.3 fit best on 2023-24 past
props. Edge = chance x decimal price - 1 at the best regulated book's price. A pick needs MIN_EDGE.

Backtest (experiments/props_value.py, 2024-25 and 2025-26): betting every positive-edge side broke
even, +0.1% +/- 2.6% over 1,541 bets. So these are tracked, not proven.

Logged to docs/lines/nba_prop_picks.csv (a later run the same day replaces that day's picks) and
graded every morning into docs/nba/prop_picks.json with tonight's list.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import propprob
from .players import NBA, STATS

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "docs" / "lines" / "nba_prop_picks.csv"
OUT = ROOT / "docs" / "nba" / "prop_picks.json"
COLS = ["date", "game_id", "athlete_id", "name", "team", "opp", "stat", "side", "line", "price", "book", "chance",
        "model_chance", "market_chance", "edge", "projection"]
BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
         "betrivers": "BetRivers", "fanatics": "Fanatics", "espnbet": "ESPN BET", "hardrockbet": "Hard Rock",
         "ballybet": "Bally Bet", "pointsbetus": "PointsBet", "unibet_us": "Unibet", "betparx": "betPARX"}


def _dec(p):
    return 1 + (p / 100 if p > 0 else 100 / -p)


def params():
    d = json.loads(propprob.PARAMS.read_text())
    return d["k"], d["w"], d["min_edge"]


def picks(day, todays):
    """todays: {game_id: {"home": [player rows], "away": [...]}} with row["props"] attached."""
    k, w, min_edge = params()
    rows = []
    for gid, e in todays.items():
        for side in ("home", "away"):
            for r in e.get(side, []):
                for s, p in (r.get("props") or {}).items():
                    if s not in STATS or not p.get("best_over") or not p.get("best_under") or not p.get("over") or not p.get("under"):
                        continue
                    po, pu, pp = (float(v[0]) for v in propprob.chances(np.array([r[s]]), np.array([p["line"]]), k[s]))
                    model = po / max(1 - pp, 1e-9)                       # chance of over, given no push
                    a, b = 1 / _dec(p["over"]), 1 / _dec(p["under"])
                    market = a / (a + b)
                    ch = w * model + (1 - w) * market
                    for sd, c, price, book in (("over", ch, p["best_over"], p.get("best_over_book")),
                                               ("under", 1 - ch, p["best_under"], p.get("best_under_book"))):
                        ev = c * _dec(price) - 1
                        if ev >= min_edge:
                            rows.append({"date": day.isoformat(), "game_id": str(gid), "athlete_id": r["id"], "name": r["name"],
                                         "team": r["team"], "opp": r["opp"], "stat": s, "side": sd, "line": p["line"],
                                         "price": int(price), "book": BOOKS.get(book, book), "chance": round(c, 3),
                                         "model_chance": round(model if sd == "over" else 1 - model, 3),
                                         "market_chance": round(market if sd == "over" else 1 - market, 3),
                                         "edge": round(ev, 3), "projection": round(float(r[s]), 1)})
    return sorted(rows, key=lambda x: -x["edge"])


def log(day, rows):
    new = pd.DataFrame(rows, columns=COLS)
    if LOG.exists():
        old = pd.read_csv(LOG, dtype={"game_id": str})
        old = old[old.date != day.isoformat()]
        new = pd.concat([old, new], ignore_index=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(LOG, index=False)
    return len(rows)


def _record(x):
    n = len(x)
    if not n:
        return {"picks": 0}
    w, l, p = int((x.res > 0).sum()), int((x.res < 0).sum()), int((x.res == 0).sum())
    return {"picks": n, "w": w, "l": l, "p": p, "units": round(float(x.profit.sum()), 2),
            "roi": round(100 * float(x.profit.mean()), 1)}


def grade(season, day, tonight):
    """Grade logged picks against the box scores; write tonight's list and the record."""
    out = {"generated": day.isoformat(), "tonight": tonight, "record": None,
           "method": "Chance = 30% model + 70% market (no vig); a pick needs 3%+ edge at the best US-regulated price.",
           "backtest": "Past props, 2024-25 and 2025-26: every positive-edge side broke even, +0.1% ± 2.6% over 1,541 bets. Tracked, not proven."}
    if LOG.exists():
        lg = pd.read_csv(LOG, dtype={"game_id": str})
        b = NBA.box(season, refresh=False)
        b = b.melt(id_vars=["game_id", "athlete_id"], value_vars=STATS, var_name="stat", value_name="actual")
        b["game_id"] = b.game_id.astype(str)
        m = lg.merge(b, on=["game_id", "athlete_id", "stat"], how="inner")
        if len(m):
            m["res"] = np.where(m.side == "over", np.sign(m.actual - m.line), np.sign(m.line - m.actual))
            m["profit"] = np.where(m.res > 0, m.price.map(_dec) - 1, np.where(m.res < 0, -1.0, 0.0))
            last = m.date.max()
            out["record"] = {"season": _record(m), "last_night": _record(m[m.date == last]), "last_night_date": last,
                             "by_stat": {s: _record(g) for s, g in m.groupby("stat")},
                             "last_night_picks": m[m.date == last][["name", "team", "stat", "side", "line", "price", "book",
                                                                   "edge", "actual", "res"]].to_dict("records")}
    OUT.write_text(json.dumps(out, indent=1, allow_nan=False, default=float))
    return out
