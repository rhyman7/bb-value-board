"""Do the NBA player projections beat past prop lines?

Lines: data/odds/nba_props.csv.gz (oddsapi/prop_history.py), every US book 60 minutes before tip,
reduced to one consensus line per player and stat (median line, median prices at it). Projections:
the walk-forward backtest in nbamodel/players.py with the live parameters (tuned on 2021-23, so
2023-24 on is out of sample). The backtest knows which regulars sat, which the line an hour before
tip mostly knows too.

Bet the model's side of every line (over if the projection is above it). Pushes are dropped.
Reported: win rate, return at the real prices, and the same for bigger gaps between projection and
line. Also the season average's side, to see whether the model adds anything beyond it.

    python -m experiments.props_backtest        # writes docs/nba/props_backtest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nbamodel import players as P
from oddsapi.prop_history import PATH
from oddsapi.props import MARKETS, _norm

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "props_backtest.json"
GAPS = {"points": (0, 1, 2, 3), "rebounds": (0, 0.5, 1, 1.5), "assists": (0, 0.5, 1, 1.5), "threes": (0, 0.3, 0.6, 0.9)}


def payout(price):
    return np.where(price > 0, price / 100.0, 100.0 / np.abs(price))


def consensus():
    q = pd.read_csv(PATH).dropna(subset=["player", "line"])
    q["stat"] = q.market.map(MARKETS)
    q["date"] = pd.to_datetime(q.commence_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    q["who"] = q.player.map(_norm)
    out = []
    for (date, who, stat), g in q.groupby(["date", "who", "stat"]):
        line = g.line.median()
        at = g[g.line == line]
        if at.empty:            # median between two lines: use the more common one
            line = g.line.mode().iloc[0]
            at = g[g.line == line]
        out.append({"date": date, "who": who, "stat": stat, "line": line, "over": at.over.median(),
                    "under": at.under.median(), "books": len(g)})
    return pd.DataFrame(out)


def summ(d):
    if d.empty:
        return {"n": 0}
    win = d.win.mean()
    n = len(d)
    return {"n": int(n), "win": round(float(win), 4), "se": round(float(np.sqrt(win * (1 - win) / n)), 4),
            "roi": round(float(d.ret.mean()), 4), "roi_se": round(float(d.ret.std() / np.sqrt(n)), 4)}


def bets(m, col):
    m = m.copy()
    m["side_over"] = m[col] > m.line
    m = m[(m[col] != m.line) & (m.actual != m.line)]
    over_won = m.actual > m.line
    m["win"] = np.where(m.side_over, over_won, ~over_won)
    price = np.where(m.side_over, m.over, m.under)
    m["ret"] = np.where(m.win, payout(price), -1.0)
    m["gap"] = (m[col] - m.line).abs()
    return m


def main():
    lines = consensus()
    params = P.load_params()
    x = P.walk([2023, 2024, 2025, 2026])
    p = P.project(x[x.season >= 2024], params)
    p = p[p.a_minutes.notna()].copy()
    p["date"] = pd.to_datetime(p.date).dt.strftime("%Y-%m-%d")
    p["who"] = p.name.map(_norm)
    rows = []
    for s in P.STATS:
        q = p[["date", "who", "season", f"p_{s}", f"avg_{s}", f"a_{s}"]].rename(
            columns={f"p_{s}": "proj", f"avg_{s}": "avg", f"a_{s}": "actual"})
        q["stat"] = s
        rows.append(q)
    proj = pd.concat(rows)
    m = lines.merge(proj, on=["date", "who", "stat"], how="inner")
    res = {"lines": int(len(lines)), "matched": int(len(m)), "match_rate": round(len(m) / max(len(lines), 1), 3),
           "games_days": int(lines.date.nunique()), "break_even_at_-110": 0.5238}
    mod, avg = bets(m, "proj"), bets(m.dropna(subset=["avg"]), "avg")
    res["model_all"] = summ(mod)
    res["season_avg_all"] = summ(avg)
    res["by_stat"] = {}
    for s in P.STATS:
        ms = mod[mod.stat == s]
        res["by_stat"][s] = {"model": summ(ms), "season_avg": summ(avg[avg.stat == s]),
                             "by_gap": {f"{g}+": summ(ms[ms.gap >= g]) for g in GAPS[s]},
                             "over": summ(ms[ms.side_over]), "under": summ(ms[~ms.side_over]),
                             "line_mae": round(float((m[m.stat == s].actual - m[m.stat == s].line).abs().mean()), 3),
                             "model_mae": round(float((m[m.stat == s].actual - m[m.stat == s].proj).abs().mean()), 3)}
    res["by_season"] = {int(k): summ(g) for k, g in mod.groupby("season")}
    print(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
