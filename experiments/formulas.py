"""Other formulas on the same NBA data: do any pick the side better against the closing line?

Every model is walk-forward: a season's model is fitted only on earlier seasons, and every feature
uses only games before that game's date. Graded on 2021-23 (tuning) and 2024-26 (check) against
consensus closing spreads, plus the average miss on the margin.

Formulas:
  current      the board's model (team ratings, market ratings, rest, missing players, impact)
  four_factors Dean Oliver's four factors (shooting, turnovers, offensive rebounds, free throws) for
               each team's offense and defense, recency-weighted, weights learned by regression
  elo          margin-of-victory Elo (538-style), carried over seasons at 75%
  form         recency-weighted raw point margin, no opponent adjustment
  all_linear   every input above in one ridge regression (learned weights for each)
  boosted      gradient-boosted trees on every input (finds nonlinear combinations)
  ensemble     average of current, all_linear and boosted
  stacked      regression weights on current, four_factors, elo and form, learned on earlier seasons

    python -m experiments.formulas            # writes docs/nba/formula_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nbamodel import backtest, data
from nbamodel.formulas import FF, elo, gbm, prior_form, ridge, team_games

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "formula_test.json"
TUNE, CHECK = (2021, 2023), (2024, 2026)


def build(use_availability=True):
    res = backtest.run(2027, verbose=False, use_availability=use_availability)
    bt, rated = res["games"], res["rated"]
    seasons = sorted(rated.season.unique())
    f = prior_form(team_games([seasons[0] - 1] + seasons, data))
    g = rated.copy()
    g["game_id"] = g.game_id.astype(str)
    for side, idc in (("h", "home_id"), ("a", "away_id")):
        p = f.drop(columns=["season", "date"]).rename(columns={"team_id": idc})
        p.columns = [c if c in ("game_id", idc) else f"{c}_{side}" for c in p.columns]
        g = g.merge(p.assign(**{idc: p[idc].astype(int)}), on=["game_id", idc], how="left")
    for k in ["efg", "tov", "orb", "ftr"]:
        g[f"{k}_off_diff"] = g[f"p_{k}_o_h"] - g[f"p_{k}_d_a"] - (g[f"p_{k}_o_a"] - g[f"p_{k}_d_h"])
        g[f"{k}_net"] = (g[f"p_{k}_o_h"] - g[f"p_{k}_d_h"]) - (g[f"p_{k}_o_a"] - g[f"p_{k}_d_a"])
    g["form_diff"] = g.p_mov_h - g.p_mov_a
    g["elo_diff"] = g.game_id.map(elo(g.assign(home_id=g.home_id.astype(int), away_id=g.away_id.astype(int))))
    g["mk_fill"] = g.mk_margin.fillna(g.r_margin)
    g = g.merge(bt[["game_id", "m_margin"]].assign(game_id=lambda d: d.game_id.astype(str)), on="game_id", how="left")
    return g


BASE = ["rest_diff", "b2b_h", "b2b_a", "g7_diff"]


def walk(g, cols, kind="ridge", lam=10.0, first=2018):
    out = pd.Series(np.nan, index=g.index)
    for s in range(first, int(g.season.max()) + 1):
        tr = g[(g.season < s)].dropna(subset=cols + ["margin"])
        te = g[g.season == s].dropna(subset=cols)
        if len(tr) < 500 or not len(te):
            continue
        if kind == "ridge":
            out[te.index] = ridge(tr[cols].to_numpy(float), tr.margin.to_numpy(float), lam)(te[cols].to_numpy(float))
        else:
            m = gbm()
            m.fit(tr[cols].to_numpy(float), tr.margin.to_numpy(float))
            out[te.index] = m.predict(te[cols].to_numpy(float))
    return out


def grade(g, col):
    x = g[g.season.between(TUNE[0], CHECK[1]) & g.close_spread.notna() & g[col].notna() & g.m_margin.notna()]
    out = {}
    for name, (a, b) in (("tune", TUNE), ("check", CHECK)):
        s = x[x.season.between(a, b)]
        d = (s.margin - s[col]).abs() - (s.margin - s.m_margin).abs()
        row = {"games": int(len(s)), "mae": round(float((s.margin - s[col]).abs().mean()), 3),
               "mae_vs_current": round(float(d.mean()), 3), "se": round(float(d.std() / np.sqrt(len(d))), 3)}
        for k in (0, 1, 2, 3):
            gap = s[col] - s.close_spread
            ss = s[gap.abs() >= k]
            r = np.sign(ss.margin - ss.close_spread) * np.sign(ss[col] - ss.close_spread)
            w, l = int((r > 0).sum()), int((r < 0).sum())
            row[f"ats_{k}+"] = {"w": w, "l": l, "pct": round(100 * w / (w + l), 1) if w + l else None,
                                "se": round(100 * (0.25 / (w + l)) ** 0.5, 1) if w + l else None}
        out[name] = row
    out["close_mae"] = {n: round(float((x[x.season.between(a, b)].margin - x[x.season.between(a, b)].close_spread).abs().mean()), 3)
                        for n, (a, b) in (("tune", TUNE), ("check", CHECK))}
    return out


def main(use_availability=True):
    g = build(use_availability)
    have = [c for c in ["miss_diff", "imp_diff"] if c in g and g[c].notna().any()]
    core = ["r_margin", "mk_fill", "has_mk"] + BASE + have
    g["has_mk"] = g.has_mk.astype(float)
    g["current"] = g.m_margin
    g["four_factors"] = walk(g, FF + BASE + have)
    g["elo"] = walk(g, ["elo_diff"] + BASE + have)
    g["form"] = walk(g, ["form_diff"] + BASE + have)
    allc = core + FF + ["elo_diff", "form_diff", "r_pace"]
    g["all_linear"] = walk(g, allc)
    g["boosted"] = walk(g, allc, kind="gbm")
    g["ensemble"] = g[["current", "all_linear", "boosted"]].mean(axis=1, skipna=False)
    g["stacked"] = walk(g, ["current", "four_factors", "elo", "form"], lam=1.0, first=2019)
    res = {"availability": use_availability}
    for c in ["current", "four_factors", "elo", "form", "all_linear", "boosted", "ensemble", "stacked"]:
        res[c] = grade(g, c)
        print(c, json.dumps({k: (v.get("mae"), v.get("mae_vs_current"), v.get("ats_0+", {}).get("pct"), v.get("ats_2+", {}).get("pct"))
                             for k, v in res[c].items() if k in ("tune", "check")}), flush=True)
    return res, g


if __name__ == "__main__":
    out = {}
    for av in (True, False):
        out["with_who_played" if av else "pregame_only"], _ = main(av)
    OUT.write_text(json.dumps(out, indent=1))
