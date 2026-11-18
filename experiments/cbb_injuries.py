"""Can we tell who is out tonight in college without an injury report?

Rule tested: a regular (30%+ of the team's last 15 games, 10+ minutes when he plays) who has missed
the team's last k games in a row is out tonight. For each k: how often the rule is right, and what
it does to the player projections, against knowing nothing (everyone available) and knowing exactly
who sat (the backtest's usual assumption).

    python -m experiments.cbb_injuries      # writes docs/cbb/injury_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cbbmodel.players import CBB
from nbamodel import players as P

KS = [1, 2, 3]
SEASONS = [2022, 2023, 2024, 2025, 2026]     # 2022 warms up the prior; graded 2023-26 like the projections
OUT = Path(__file__).resolve().parent.parent / "docs" / "cbb" / "injury_test.json"


def run():
    rows, avail = [], []
    prev = None
    for season in SEASONS:
        b = CBB.box(season)
        st = P.State(prev, CBB.team_min)
        for day, todays in b.groupby("date"):
            for gid, gp in todays.groupby("game_id"):
                played = set(gp.athlete_id)
                actual = gp.set_index("athlete_id")
                for tid, tp in gp.groupby("team_id"):
                    opp = int(tp.opponent_team_id.iloc[0])
                    pool = st.pool(tid)
                    if not pool:
                        continue
                    stk = st.streak(tid)
                    for a, (role, share) in pool.items():
                        if P.regular(role, share):
                            avail.append((season, min(stk[a], 6), a in played, role))
                    modes = {"none": set(), "truth": {a for a in pool if a not in played}}
                    for k in KS:
                        modes[f"miss{k}"] = P.inferred_out(st, tid, k)
                    for mode, out_ids in modes.items():
                        for r in P.team_rows(st, tid, opp, out_ids):
                            a = r["athlete_id"]
                            if a not in played:
                                continue           # graded on players who played, as everywhere else
                            r.update({"season": season, "game_id": gid, "mode": mode,
                                      **{f"a_{s}": float(actual.at[a, s]) for s in P.STATS + ["minutes"]}})
                            r.pop("out")
                            rows.append(r)
            for gid, gp in todays.groupby("game_id"):
                st.add_game(gp)
        prev = st.carry()
        print(season, len(rows), flush=True)
    x = pd.DataFrame(rows)
    av = pd.DataFrame(avail, columns=["season", "streak", "played", "role"])
    av = av[av.season >= 2023]
    params = P.load_params(CBB)
    res = {"rule": "regular who missed the team's last k games in a row is out", "seasons": "2023-2026", "by_streak": {}, "modes": {}}
    for k, g in av.groupby("streak"):
        res["by_streak"][int(k)] = {"regular_team_games": int(len(g)), "played_pct": round(100 * g.played.mean(), 1)}
    for k in KS:
        flag = av.streak >= k
        res[f"miss{k}_rule"] = {"flagged": int(flag.sum()), "flagged_sat_pct": round(100 * (~av.played[flag]).mean(), 1),
                                "absences_caught_pct": round(100 * flag[~av.played].mean(), 1)}
    keep = x[x.season >= 2023]
    # compare on the same player-games: those every mode projected (a flagged player who then
    # played has no projection under that rule; that cost is counted separately)
    key = ["game_id", "athlete_id"]
    common = None
    for mode, g in keep.groupby("mode"):
        k = set(map(tuple, g[key].to_numpy()))
        common = k if common is None else common & k
    keep = keep[[t in common for t in map(tuple, keep[key].to_numpy())]]
    for mode, g in keep.groupby("mode"):
        res["modes"][mode] = P.grade(g, params, CBB)
    for k in KS:
        res[f"miss{k}_rule"]["played_anyway_no_projection"] = int(((av.streak >= k) & av.played).sum())
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "modes"}, indent=1))
    for m, v in res["modes"].items():
        print(m, v["player_games"], {s: v[s]["model_mae"] for s in ["minutes", "points", "rebounds", "assists"]})
    return res


if __name__ == "__main__":
    run()
