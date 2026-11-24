"""The same alternative formulas as experiments/formulas.py, on the college model's data.

Graded on home-court games against ESPN's spread (neutral-site spreads are unreliable, see
cbbmodel/backtest.py). The college model's own predictions start in 2018, so the alternatives
are first fitted on 2018: 2019 is the tuning season and 2020-23 the check (the stacked blend
needs a season of the others first, so it is graded on 2020-23 only).

    python -m experiments.formulas_cbb        # writes docs/cbb/formula_test.json
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path


from cbbmodel import backtest as cbt, data as cdata
from experiments import formulas as F
from nbamodel.formulas import FF, elo, prior_form, team_games

OUT = Path(__file__).resolve().parent.parent / "docs" / "cbb" / "formula_test.json"


def build():
    d = date.today()
    last = d.year + 1 if d.month >= 8 else d.year
    bt = cbt.run(last, verbose=False)["games"]
    g = bt.copy()
    g["game_id"] = g.game_id.astype(str)
    f = prior_form(team_games(range(cbt.FIRST_RATED - 1, last + 1), src=cdata, types=(2, 3)))
    for side, idc in (("h", "home_id"), ("a", "away_id")):
        p = f.drop(columns=["season", "date"]).rename(columns={"team_id": idc})
        p.columns = [c if c in ("game_id", idc) else f"{c}_{side}" for c in p.columns]
        g[idc] = g[idc].astype(int)
        g = g.merge(p, on=["game_id", idc], how="left")
    for k in ["efg", "tov", "orb", "ftr"]:
        g[f"{k}_off_diff"] = g[f"p_{k}_o_h"] - g[f"p_{k}_d_a"] - (g[f"p_{k}_o_a"] - g[f"p_{k}_d_h"])
        g[f"{k}_net"] = (g[f"p_{k}_o_h"] - g[f"p_{k}_d_h"]) - (g[f"p_{k}_o_a"] - g[f"p_{k}_d_a"])
    g["form_diff"] = g.p_mov_h - g.p_mov_a
    g["elo_diff"] = g.game_id.map(elo(g))
    g["home"] = (~g.neutral.astype(bool)).astype(float)
    return g[g.season < last]


def main():
    g = build()
    F.TUNE, F.CHECK = (2019, 2019), (2020, 2023)
    base = ["home", "rest_diff", "b2b_h", "b2b_a", "far_road_diff", "miss_diff", "lineup_diff"]
    g["current"] = g.m_margin
    g["four_factors"] = F.walk(g, FF + base)
    g["elo"] = F.walk(g, ["elo_diff"] + base)
    g["form"] = F.walk(g, ["form_diff"] + base)
    allc = ["r_margin", "r_early"] + base + FF + ["elo_diff", "form_diff"]
    g["all_linear"] = F.walk(g, allc)
    g["boosted"] = F.walk(g, allc, kind="gbm")
    g["ensemble"] = g[["current", "all_linear", "boosted"]].mean(axis=1, skipna=False)
    g["stacked"] = F.walk(g, ["current", "four_factors", "elo", "form"], lam=1.0, first=2020)
    h = g[~g.neutral.astype(bool)]
    res = {}
    for c in ["current", "four_factors", "elo", "form", "all_linear", "boosted", "ensemble", "stacked"]:
        res[c] = F.grade(h, c)
        print(c, json.dumps({k: (v.get("mae"), v.get("mae_vs_current"), v.get("se"), v.get("ats_0+", {}).get("pct"), v.get("ats_2+", {}).get("pct"))
                             for k, v in res[c].items() if k in ("tune", "check")}), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
