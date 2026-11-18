"""Player projections: does what the opponent allows to the player's position (guard, forward,
center, as ESPN lists him) beat what the opponent allows overall?

Per stat, the projection's opponent term becomes  team_factor ** beta * (position_factor /
team_factor) ** gamma.  gamma = 0 is today's team-wide factor; beta = gamma = 1 is the position
factor alone. beta and gamma are picked on the tuning seasons, then the final calibration is refit,
and the holdout is compared player-game by player-game against today's model.

    python -m experiments.position_opp            # NBA, writes docs/nba/position_opp_test.json
    python -m experiments.position_opp cbb        # college, writes docs/cbb/position_opp_test.json
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from nbamodel import players as P

GRID = (0.0, 0.5, 1.0, 1.5)


def with_pos(x, beta, gamma):
    """A copy whose opp_ columns are the whole tuned opponent term (projected with beta = 1)."""
    y = x.copy()
    for s in P.STATS:
        y[f"opp_{s}"] = x[f"opp_{s}"] ** beta[s] * (x[f"opp_pos_{s}"] / x[f"opp_{s}"]) ** gamma[s]
    return y


def tune_gamma(x, lg):
    t = x[x.season.isin(lg.tune)]
    base = P.tune(x, lg)
    p0 = P.graded(P.project(t, {**base, "beta": {s: 0.0 for s in P.STATS}, "cal": {}}, lg))
    gamma, beta = {}, {}
    for s in P.STATS:
        best = None
        for b in GRID:
            for g in GRID:
                pred = p0[f"p_{s}"] * p0[f"opp_{s}"] ** b * (p0[f"opp_pos_{s}"] / p0[f"opp_{s}"]) ** g \
                    * (1 + base["u"][s] * p0[f"mshare_{s}"])
                e = float((p0[f"a_{s}"] - pred).abs().mean())
                if best is None or e < best[0]:
                    best = (e, b, g)
        beta[s], gamma[s] = best[1], best[2]
    return base, beta, gamma


def recal(x, params, lg):
    p = P.graded(P.project(x[x.season.isin(lg.tune)], {**params, "cal": {}}, lg))
    cal = {}
    for s in ["minutes"] + P.STATS:
        X = np.column_stack([np.ones(len(p)), p[f"p_{s}"], p[f"avg_{s}"]])
        cal[s] = [round(float(v), 4) for v in np.linalg.lstsq(X, p[f"a_{s}"].to_numpy(float), rcond=None)[0]]
    return {**params, "cal": cal}


def compare(x0, x1, p0, p1, lg, seasons):
    a = P.graded(P.project(x0[x0.season.isin(seasons)], p0, lg))
    b = P.graded(P.project(x1[x1.season.isin(seasons)], p1, lg))
    out = {"player_games": int(len(a))}
    for s in P.STATS:
        d = (b[f"a_{s}"] - b[f"p_{s}"]).abs() - (a[f"a_{s}"] - a[f"p_{s}"]).abs()
        out[s] = {"today_mae": round(float((a[f"a_{s}"] - a[f"p_{s}"]).abs().mean()), 4),
                  "position_mae": round(float((b[f"a_{s}"] - b[f"p_{s}"]).abs().mean()), 4),
                  "change": round(float(d.mean()), 4), "se": round(float(d.std() / np.sqrt(len(d))), 4)}
    return out


def main(league="nba"):
    if league == "cbb":
        from cbbmodel.players import CBB as lg
    else:
        lg = P.NBA
    seasons = list(range(min(lg.tune) - 1, max(lg.grade) + 1))
    x = P.walk(seasons, lg=lg)
    known = float((x.pos != "").mean())
    base, beta, gamma = tune_gamma(x, lg)
    x1 = with_pos(x, beta, gamma)
    new = recal(x1, {**base, "beta": {s: 1.0 for s in P.STATS}}, lg)
    # the factors themselves: how far position moves the opponent number
    spread = {s: round(float((np.log(x[f"opp_pos_{s}"]) - np.log(x[f"opp_{s}"])).std()), 4) for s in P.STATS}
    res = {"league": league, "position_known_share": round(known, 3),
           "spread_of_position_vs_team_factor_log_sd": spread,
           "today_beta": base["beta"], "new_beta": beta, "gamma": gamma,
           "tune": compare(x, x1, base, new, lg, lg.tune),
           "holdout": compare(x, x1, base, new, lg, lg.grade)}
    print(json.dumps(res, indent=1))
    (lg.docs / "position_opp_test.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["nba"]))
