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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nbamodel import backtest, data

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "formula_test.json"
TUNE, CHECK = (2021, 2023), (2024, 2026)
HL_GAMES = 20          # recency half-life for the four factors and form, in team games


def team_games(seasons, src=data, types=(2,)):
    """One row per team per game with that game's four factors for and against."""
    parts = []
    for s in seasons:
        try:
            tb = src.team_box(s)
        except Exception:
            continue
        tb = tb[tb.season_type.isin(types)].copy()
        for c in ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "free_throws_made",
                  "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "total_turnovers", "turnovers", "team_score"]:
            tb[c] = pd.to_numeric(tb[c], errors="coerce").fillna(0)
        tb["tov"] = tb.total_turnovers.where(tb.total_turnovers > 0, tb.turnovers)
        parts.append(tb)
    tb = pd.concat(parts, ignore_index=True)
    opp = tb[["game_id", "team_id", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made",
              "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "tov", "team_score"]]
    t = tb.merge(opp, left_on=["game_id", "opponent_team_id"], right_on=["game_id", "team_id"], suffixes=("", "_o"))
    poss = lambda fga, orb, tov, fta: fga - orb + tov + 0.44 * fta
    t["poss"] = 0.5 * (poss(t.field_goals_attempted, t.offensive_rebounds, t.tov, t.free_throws_attempted)
                       + poss(t.field_goals_attempted_o, t.offensive_rebounds_o, t.tov_o, t.free_throws_attempted_o))
    f = pd.DataFrame({"game_id": t.game_id.astype(str), "team_id": t.team_id.astype(int), "season": t.season,
                      "date": pd.to_datetime(t.game_date)})
    for side, sfx in (("o", ""), ("d", "_o")):
        fga = t["field_goals_attempted" + sfx].replace(0, np.nan)
        f[f"efg_{side}"] = (t["field_goals_made" + sfx] + 0.5 * t["three_point_field_goals_made" + sfx]) / fga
        f[f"tov_{side}"] = t["tov" + sfx] / t.poss
        other = "_o" if sfx == "" else ""
        f[f"orb_{side}"] = t["offensive_rebounds" + sfx] / (t["offensive_rebounds" + sfx] + t["defensive_rebounds" + other]).replace(0, np.nan)
        f[f"ftr_{side}"] = t["free_throws_made" + sfx] / fga
    f["mov"] = t.team_score - t.team_score_o
    f["pace"] = t.poss
    return f.sort_values(["team_id", "date"])


def prior_form(f):
    """Each team's recency-weighted averages from games before this one."""
    cols = [c for c in f.columns if c.endswith(("_o", "_d"))] + ["mov", "pace"]
    g = f.groupby("team_id")[cols]
    pre = g.transform(lambda s: s.shift().ewm(halflife=HL_GAMES, min_periods=5).mean())
    return pd.concat([f[["game_id", "team_id", "season", "date"]], pre.add_prefix("p_")], axis=1)


def elo(games):
    """Margin-of-victory Elo; returns home minus away rating (with home court) before each game."""
    r, out, last = {}, {}, None
    for x in games.sort_values("date").itertuples():
        if last is not None and x.season != last:
            r = {k: 1505 + 0.75 * (v - 1505) for k, v in r.items()}
        last = x.season
        h, a = r.get(x.home_id, 1505), r.get(x.away_id, 1505)
        d = h + (0 if getattr(x, "neutral", False) else 70) - a
        out[x.game_id] = d
        exp = 1 / (1 + 10 ** (-d / 400))
        win = 1.0 if x.margin > 0 else 0.0
        mult = (abs(x.margin) + 3) ** 0.8 / (7.5 + 0.006 * (d if x.margin > 0 else -d))
        k = 20 * mult * (win - exp)
        r[x.home_id], r[x.away_id] = h + k, a - k
    return pd.Series(out, name="elo_diff")


def build(use_availability=True):
    res = backtest.run(2027, verbose=False, use_availability=use_availability)
    bt, rated = res["games"], res["rated"]
    seasons = sorted(rated.season.unique())
    f = prior_form(team_games([seasons[0] - 1] + seasons))
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


FF = ["efg_off_diff", "tov_off_diff", "orb_off_diff", "ftr_off_diff", "efg_net", "tov_net", "orb_net", "ftr_net"]
BASE = ["rest_diff", "b2b_h", "b2b_a", "g7_diff"]


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    P = np.eye(Z.shape[1]) * lam
    P[0, 0] = 0
    b = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ b


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
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                              min_samples_leaf=80, l2_regularization=1.0, random_state=0)
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
