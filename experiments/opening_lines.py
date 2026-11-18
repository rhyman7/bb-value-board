"""The opening-line test on past seasons, with lines from the Odds API (oddsapi/history.py).

For each game: the opener is the consensus (median across US books) at the first daily snapshot that
had the game, at most about a day before tip; the close is hoopR's consensus close (NBA) or ESPN's
line (college). Model numbers are the walk-forward backtests, fit on earlier seasons only.

NBA is run twice: the standard backtest knows which players sat (it reads the box score), which the
market did not know when the opener was posted, so it flatters the model; the second run leaves
missing players and market ratings out, which is closer to what the model knew at the open.

    python -m experiments.opening_lines      # writes docs/opening_lines_test.json
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ODDS = ROOT / "data" / "odds"
OUT = ROOT / "docs" / "opening_lines_test.json"
NBA_ALIAS = {"Los Angeles Clippers": "LA Clippers"}


def odds(key):
    p = ODDS / f"{key}.csv.gz"
    with gzip.open(p, "rt") as f:
        d = pd.read_csv(f)
    d = d.dropna(subset=["event_id"])
    d["snap"] = pd.to_datetime(d.snapshot, utc=True)
    g = d.groupby(["event_id", "snap"]).agg(home=("home_team", "first"), away=("away_team", "first"),
                                            tip=("commence_time", "first"), spread=("home_spread", "median"),
                                            total=("total", "median"), books=("book", "nunique")).reset_index()
    g = g.dropna(subset=["spread"]).sort_values("snap")
    first = g.groupby("event_id").first().reset_index()
    last = g.groupby("event_id").last().reset_index()[["event_id", "spread", "total", "snap"]].rename(
        columns={"spread": "morning_spread", "total": "morning_total", "snap": "morning_snap"})
    o = first.merge(last, on="event_id")
    o["tip"] = pd.to_datetime(o.tip, utc=True)
    o["date"] = o.tip.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    o["hours_before"] = (o.tip - o.snap).dt.total_seconds() / 3600
    # the Odds API gives the home team's handicap; our lines are home minus away
    o["open_margin"] = -o.spread
    o["morning_margin"] = -o.morning_spread
    return o.rename(columns={"total": "open_total"})


def _norm(name):
    n = name.lower().replace("&", " and ").replace("univ.", "").replace("-", " ").replace(".", "").replace("'", "")
    n = re.sub(r"\bst\b", "state", n)
    return re.sub(r"\s+", " ", n).strip()


def match(o, sched, alias=None):
    """Join to schedule game ids on date and team names. Names are normalized, and a game also matches
    when only one side's name agrees, since each team plays once a day."""
    alias = alias or {}
    s = sched.copy()
    s["date"] = pd.to_datetime(s.date, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    s = s[["game_id", "date", "home_display_name", "away_display_name"]].assign(
        h=lambda d: d.home_display_name.map(_norm), a=lambda d: d.away_display_name.map(_norm))
    if pd.api.types.is_numeric_dtype(s.game_id):
        s["game_id"] = s.game_id.astype("int64")
    s["game_id"] = s.game_id.astype(str)
    o = o.assign(h=o.home.map(lambda x: _norm(alias.get(x, x))), a=o.away.map(lambda x: _norm(alias.get(x, x))))
    o["game_id"] = None
    for keys in (["h", "a"], ["h"], ["a"]):
        need = o.game_id.isna()
        ss = s.drop_duplicates(["date"] + keys, keep=False)
        hit = o[need].merge(ss[["date"] + keys + ["game_id"]], on=["date"] + keys, how="left", suffixes=("_old", ""))
        o.loc[need, "game_id"] = hit.game_id.to_numpy()
    o["game_id"] = o.game_id.astype("string")
    return o


def record(model, line, actual, gap=0.0):
    lean = model - line
    sel = lean.abs() >= gap if gap else lean.abs() > 0
    res = np.sign(actual - line)[sel] * np.sign(lean[sel])
    w, l = int((res > 0).sum()), int((res < 0).sum())
    n = w + l
    return {"w": w, "l": l, "p": int((res == 0).sum()), "pct": round(100 * w / n, 1) if n else None,
            "roi": round(100 * (w * 100 / 110 - l) / n, 1) if n else None, "se": round(100 * (0.25 / n) ** 0.5, 1) if n else None}


def grade(x, mcol, ocol, ccol, acol):
    x = x.dropna(subset=[mcol, ocol, acol])
    out = {"games": int(len(x)),
           "mae": {"model": round(float((x[acol] - x[mcol]).abs().mean()), 2), "opener": round(float((x[acol] - x[ocol]).abs().mean()), 2)},
           "vs_opener": {f"{k}+": record(x[mcol], x[ocol], x[acol], k) for k in [0, 1, 2, 3, 4]}}
    c = x.dropna(subset=[ccol]) if ccol else x.iloc[0:0]
    if len(c):
        out["mae"]["close"] = round(float((c[acol] - c[ccol]).abs().mean()), 2)
        mv = (c[ccol] - c[ocol])
        lean = (c[mcol] - c[ocol])
        moved = (mv != 0) & (lean != 0)
        out["line_moved_toward_model_pct"] = round(100 * float((np.sign(mv[moved]) == np.sign(lean[moved])).mean()), 1)
        out["games_line_moved"] = int(moved.sum())
        out["vs_close_at_open_side"] = {f"{k}+": record(c[mcol], c[ocol], c[ccol], k) for k in [1, 2, 3]}
    return out


def nba(res):
    from nbamodel import backtest, data
    o = pd.concat([odds(f"nba_{s}").assign(season=s) for s in (2025, 2026)], ignore_index=True)
    sched = pd.concat([data.schedule(s) for s in (2025, 2026)], ignore_index=True)
    m = match(o, sched, NBA_ALIAS)
    res["nba_matched"] = f"{int(m.game_id.notna().sum())} of {len(m)}"
    m = m.dropna(subset=["game_id"])
    for label, avail in (("nba_standard_model", True), ("nba_without_injury_hindsight", False)):
        bt = backtest.run(2026, verbose=False, use_availability=avail)["games"]
        bt["game_id"] = bt.game_id.astype(str)
        x = m.merge(bt[["game_id", "season", "margin", "total", "m_margin", "m_total", "close_spread", "close_total"]], on="game_id")
        res[label] = {"side": grade(x, "m_margin", "open_margin", "close_spread", "margin"),
                      "total": grade(x, "m_total", "open_total", "close_total", "total"),
                      "median_hours_before_tip": round(float(x.hours_before.median()), 1)}
        print(label, json.dumps(res[label]["side"]["vs_opener"]["0+"]), flush=True)


def cbb(res):
    from cbbmodel import backtest, data
    o = pd.concat([odds(f"cbb_{s}").assign(season=s) for s in (2022, 2023)], ignore_index=True)
    sched = pd.concat([data.schedule(s) for s in (2022, 2023)], ignore_index=True)
    m = match(o, sched)
    res["cbb_matched"] = f"{int(m.game_id.notna().sum())} of {len(m)}"
    res["cbb_unmatched_names"] = sorted(set(m[m.game_id.isna()].home))[:40]
    m = m.dropna(subset=["game_id"])
    bt = backtest.run(2023, verbose=False)["games"]
    bt["game_id"] = bt.game_id.astype(str)
    x = m.merge(bt[["game_id", "season", "neutral", "margin", "total", "m_margin", "r_total", "close_spread"]], on="game_id")
    x = x[~x.neutral]
    res["cbb_model"] = {"side": grade(x, "m_margin", "open_margin", "close_spread", "margin"),
                        "total": grade(x, "r_total", "open_total", None, "total"),
                        "median_hours_before_tip": round(float(x.hours_before.median()), 1), "note": "home-court games only (ESPN neutral lines unreliable)"}
    print("cbb", json.dumps(res["cbb_model"]["side"]["vs_opener"]["0+"]), flush=True)


if __name__ == "__main__":
    res = {}
    nba(res)
    cbb(res)
    OUT.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:4000])
