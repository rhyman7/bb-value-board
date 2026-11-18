"""College: does a box-score lineup strength help, beyond the team ratings and missing players?

For each game, each team's expected players (played in the team's last 5 games and hasn't sat the
last 2 in a row, the same rule the live board uses), weighted by their usual minutes, rated by Game
Score per 40 (this season to date plus half of last season, shrunk toward replacement). Early in a
season this leans on last season's numbers, so it also tests whether player stats beat the
preseason team prior.

    python -m experiments.cbb_lineup        # writes docs/cbb/lineup_test.json
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from cbbmodel import availability, backtest

OUT = Path(__file__).resolve().parent.parent / "docs" / "cbb" / "lineup_test.json"
REPL, PRIOR = 0.28, 150.0


def strength(seasons):
    rows = []
    prev = {}
    for s in seasons:
        pb = availability.box(s)
        pb = pb[pb.minutes > 0].sort_values(["date", "game_id"])
        tot = defaultdict(lambda: [0.0, 0.0])
        recent = defaultdict(lambda: deque(maxlen=5))          # team -> deque of set(athletes)
        usual = defaultdict(lambda: deque(maxlen=10))
        for day, todays in pb.groupby("date"):
            for gid, g in todays.groupby("game_id"):
                r = {"game_id": gid}
                sides = list(g.team_id.unique())
                vals = {}
                for t in sides:
                    games = list(recent[t])
                    if not games:
                        continue
                    cand = set().union(*games)
                    sat2 = {a for a in cand if len(games) >= 2 and a not in games[-1] and a not in games[-2]}
                    ids = [a for a in cand if a not in sat2]
                    w = np.array([np.mean(usual[(t, a)]) for a in ids])
                    rate = []
                    for a in ids:
                        g_, m_ = tot[a]
                        pg, pm = prev.get(a, (0.0, 0.0))
                        rate.append((g_ + 0.5 * pg + REPL * PRIOR) / (m_ + 0.5 * pm + PRIOR) * 40)
                    vals[t] = float(5 * (w * np.array(rate)).sum() / w.sum()) if w.sum() else np.nan
                rows.append((gid, vals))
            for x in todays.itertuples():
                t = tot[x.athlete_id]
                t[0] += x.gmsc
                t[1] += x.minutes
                usual[(x.team_id, x.athlete_id)].append(x.minutes)
            for (gid, t), g in todays.groupby(["game_id", "team_id"]):
                recent[t].append(set(g.athlete_id))
        prev = {a: (v[0], v[1]) for a, v in tot.items()}
    return rows


if __name__ == "__main__":
    from cbbmodel import data
    seasons = list(range(backtest.FIRST_RATED - 1, 2024))
    st = dict(strength(seasons))
    g = data.all_games(2023, refresh_spreads=False)
    feat = []
    for x in g.itertuples():
        v = st.get(str(x.game_id), {})
        h, a = v.get(int(x.home_id)), v.get(int(x.away_id))
        feat.append({"game_id": str(x.game_id), "lineup_diff": (h - a) if h is not None and a is not None and not (np.isnan(h) or np.isnan(a)) else 0.0})
    extra = pd.DataFrame(feat)
    base = backtest.run(2023, verbose=False)["games"]
    feats = list(backtest.FEATURES)
    try:
        backtest.FEATURES[:] = feats + ["lineup_diff"]
        new = backtest.run(2023, verbose=False, extra=extra)["games"]
    finally:
        backtest.FEATURES[:] = feats
    j = base.set_index("game_id")[["season", "neutral", "margin", "m_margin"]].join(new.set_index("game_id")[["m_margin"]], rsuffix="_new")
    j = j[~j.neutral.astype(bool)]
    res = {}
    for name, ss in (("tune", backtest.TUNE), ("holdout", backtest.GRADE)):
        x = j[j.season.isin(ss)]
        d = (x.margin - x.m_margin_new).abs() - (x.margin - x.m_margin).abs()
        res[name] = {"games": int(len(x)), "base_mae": round(float((x.margin - x.m_margin).abs().mean()), 4), "change": round(float(d.mean()), 4), "se": round(float(d.std() / np.sqrt(len(d))), 4)}
    res["passes"] = res["tune"]["change"] < 0 and res["holdout"]["change"] < -res["holdout"]["se"]
    print(json.dumps(res))
    OUT.write_text(json.dumps(res, indent=1))
