"""Team totals, tested for free: each team's line is implied by the game's spread and total
(home = (total + margin) / 2), and the model's team number the same way from its own margin and total.

This is not a real team-totals market (books price those on their own and with more hold), but it says
whether the model's view of one team's scoring beats the market's before we spend credits on real lines.

    python -m experiments.team_totals      # writes docs/team_totals_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.opening_lines import match, odds, record

OUT = Path(__file__).resolve().parent.parent / "docs" / "team_totals_test.json"
GAPS = [0, 1, 2, 3, 4]


def team_rows(x, m_margin, m_total, l_margin, l_total):
    """One row per team per game: model points, implied line, actual points."""
    rows = []
    for sign, side in ((1, "home"), (-1, "away")):
        rows.append(pd.DataFrame({
            "season": x.season.to_numpy(), "side": side,
            "model": ((x[m_total] + sign * x[m_margin]) / 2).to_numpy(),
            "line": ((x[l_total] + sign * x[l_margin]) / 2).to_numpy(),
            "actual": ((x.total + sign * x.margin) / 2).to_numpy()}))
    return pd.concat(rows, ignore_index=True).dropna()


def grade(t):
    out = {"team_games": int(len(t)),
           "mae": {"model": round(float((t.actual - t.model).abs().mean()), 2), "line": round(float((t.actual - t.line).abs().mean()), 2)},
           "by_gap": {f"{k}+": record(t.model, t.line, t.actual, k) for k in GAPS}}
    out["by_season"] = {int(s): record(g.model, g.line, g.actual, 0) for s, g in t.groupby("season")}
    return out


def nba(res):
    from nbamodel import backtest
    for label, avail in (("nba_without_injury_hindsight", False), ("nba_standard_model", True)):
        r = backtest.run(2026, verbose=False, use_availability=avail)["games"]
        x = r[(r.season >= 2021) & (r.season <= 2026)].dropna(subset=["close_spread", "close_total"])
        res[label] = grade(team_rows(x, "m_margin", "m_total", "close_spread", "close_total"))
        res[label]["line"] = "consensus close, 2021-26 seasons"
        print(label, json.dumps(res[label]["by_gap"]["0+"]), flush=True)


def cbb(res):
    from cbbmodel import backtest, data
    o = pd.concat([odds(f"cbb_{s}").assign(season=s) for s in (2022, 2023)], ignore_index=True)
    sched = pd.concat([data.schedule(s) for s in (2022, 2023)], ignore_index=True)
    m = match(o, sched).dropna(subset=["game_id"])
    bt = backtest.run(2023, verbose=False)["games"]
    bt["game_id"] = bt.game_id.astype(str)
    x = m.drop(columns=["season"]).merge(bt[["game_id", "season", "neutral", "margin", "total", "m_margin", "r_total"]], on="game_id")
    x = x[~x.neutral.astype(bool)]
    res["cbb_model"] = grade(team_rows(x, "m_margin", "r_total", "morning_margin", "morning_total"))
    res["cbb_model"]["line"] = "Odds API consensus, noon ET (about 8 hours before tip), home-court games 2022-23"
    print("cbb", json.dumps(res["cbb_model"]["by_gap"]["0+"]), flush=True)


if __name__ == "__main__":
    res = {}
    nba(res)
    cbb(res)
    OUT.write_text(json.dumps(res, indent=1))
