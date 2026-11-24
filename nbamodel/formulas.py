"""Building blocks for other game formulas: four factors, recency-weighted form and Elo.

Shared by the formula tests (experiments/formulas.py, formulas_cbb.py) and the college model's
blend (cbbmodel/blend.py). Everything here uses only games before the one it describes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HL_GAMES = 20          # recency half-life for the four factors and form, in team games
FF = ["efg_off_diff", "tov_off_diff", "orb_off_diff", "ftr_off_diff", "efg_net", "tov_net", "orb_net", "ftr_net"]
ELO_START, ELO_CARRY, ELO_HOME = 1505, 0.75, 70


def team_games(seasons, src, types=(2,)):
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


def current_form(f):
    """Each team's recency-weighted averages including every game played, for games not yet played."""
    cols = [c for c in f.columns if c.endswith(("_o", "_d"))] + ["mov", "pace"]
    cur = f.groupby("team_id")[cols].apply(lambda d: d.ewm(halflife=HL_GAMES, min_periods=5).mean().iloc[-1])
    return cur.add_prefix("p_")


def elo(games, return_state=False):
    """Margin-of-victory Elo (538-style), carried over seasons at 75%. Returns home minus away
    rating (with home court, none on a neutral floor) before each game; with return_state also the
    ratings after the last game and that game's season."""
    r, out, last = {}, {}, None
    for x in games.sort_values("date").itertuples():
        if last is not None and x.season != last:
            r = {k: ELO_START + ELO_CARRY * (v - ELO_START) for k, v in r.items()}
        last = x.season
        h, a = r.get(x.home_id, ELO_START), r.get(x.away_id, ELO_START)
        d = h + (0 if bool(getattr(x, "neutral", False)) else ELO_HOME) - a
        out[x.game_id] = d
        if pd.isna(x.margin):
            continue
        exp = 1 / (1 + 10 ** (-d / 400))
        win = 1.0 if x.margin > 0 else 0.0
        mult = (abs(x.margin) + 3) ** 0.8 / (7.5 + 0.006 * (d if x.margin > 0 else -d))
        k = 20 * mult * (win - exp)
        r[x.home_id], r[x.away_id] = h + k, a - k
    s = pd.Series(out, name="elo_diff")
    return (s, r, last) if return_state else s


def matchup(g):
    """Four-factor and form differences from the p_*_h / p_*_a columns."""
    for k in ["efg", "tov", "orb", "ftr"]:
        g[f"{k}_off_diff"] = g[f"p_{k}_o_h"] - g[f"p_{k}_d_a"] - (g[f"p_{k}_o_a"] - g[f"p_{k}_d_h"])
        g[f"{k}_net"] = (g[f"p_{k}_o_h"] - g[f"p_{k}_d_h"]) - (g[f"p_{k}_o_a"] - g[f"p_{k}_d_a"])
    g["form_diff"] = g.p_mov_h - g.p_mov_a
    return g


def attach(g, form):
    """Merge pre-game form (one row per team per game) onto games as p_*_h and p_*_a."""
    for side, idc in (("h", "home_id"), ("a", "away_id")):
        p = form.drop(columns=["season", "date"]).rename(columns={"team_id": idc})
        p.columns = [c if c in ("game_id", idc) else f"{c}_{side}" for c in p.columns]
        g = g.merge(p, on=["game_id", idc], how="left")
    return matchup(g)


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    P = np.eye(Z.shape[1]) * lam
    P[0, 0] = 0
    b = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ b


def gbm():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=80, l2_regularization=1.0, random_state=0)
