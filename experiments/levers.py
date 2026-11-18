"""Cheap tests of model levers that need no new data (keep rule as in impact_test):

- travel: kilometres each team travelled since its last game, and time zones crossed
- altitude: home games in Denver and Utah
- 3-in-4: a team's third game in four nights
- the same travel and 3-in-4 terms in the total

    python -m experiments.levers          # writes docs/nba/levers_test.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.impact_test import HOLD, TUNE, run
from nbamodel import data, model

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "levers_test.json"
ARENA = {  # lat, lon
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975), "CHA": (35.225, -80.839),
    "CHI": (41.881, -87.674), "CLE": (41.496, -81.688), "DAL": (32.790, -96.810), "DEN": (39.749, -105.008),
    "DET": (42.341, -83.055), "GS": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (33.945, -118.343), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051), "MIA": (25.781, -80.188),
    "MIL": (43.045, -87.917), "MIN": (44.979, -93.276), "NO": (29.949, -90.082), "NY": (40.751, -73.993),
    "OKC": (35.463, -97.515), "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SA": (29.427, -98.438), "TOR": (43.643, -79.379),
    "UTAH": (40.768, -111.901), "WSH": (38.898, -77.021),
}
ARENA["NOP"], ARENA["GSW"], ARENA["NYK"], ARENA["SAS"], ARENA["UTA"], ARENA["PHO"] = ARENA["NO"], ARENA["GS"], ARENA["NY"], ARENA["SA"], ARENA["UTAH"], ARENA["PHX"]
HIGH = {"DEN", "UTAH", "UTA"}


def km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    p = math.pi / 180
    h = math.sin((la2 - la1) * p / 2) ** 2 + math.cos(la1 * p) * math.cos(la2 * p) * math.sin((lo2 - lo1) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def schedule_features(g):
    g = g.sort_values(["date", "game_id"])
    last_loc, dates = {}, {}
    rows = []
    for x in g.itertuples():
        here = ARENA.get(x.home)
        r = {"game_id": x.game_id}
        for side, tid, home_code in (("h", x.home_id, x.home), ("a", x.away_id, x.away)):
            prev = last_loc.get(tid)
            r[f"km_{side}"] = km(prev, here) / 1000 if prev and here else 0.0
            r[f"tz_{side}"] = abs(prev[1] - here[1]) / 15 if prev and here else 0.0
            ds = [d for d in dates.get(tid, []) if 0 < (x.date - d).days <= 3]
            r[f"b3in4_{side}"] = float(len(ds) >= 2)
        r["alt"] = float(x.home in HIGH and x.away not in HIGH)
        rows.append(r)
        for tid in (x.home_id, x.away_id):
            if here:
                last_loc[tid] = here
            dates.setdefault(tid, []).append(x.date)
    d = pd.DataFrame(rows)
    d["km_diff"] = d.km_a - d.km_h          # + = away team travelled more
    d["tz_diff"] = d.tz_a - d.tz_h
    d["b3in4_diff"] = d.b3in4_a - d.b3in4_h
    return d


def compare_col(base, new, col):
    j = base[["season", col.replace("m_", ""), col]].join(new[[col]], rsuffix="_new", how="inner")
    act = col.replace("m_", "")
    out = {}
    for name, (a, b) in (("tune", TUNE), ("holdout", HOLD)):
        x = j[(j.season >= a) & (j.season <= b)]
        d = (x[act] - x[col + "_new"]).abs() - (x[act] - x[col]).abs()
        out[name] = {"change": round(float(d.mean()), 4), "se": round(float(d.std() / np.sqrt(len(d))), 4)}
    out["passes"] = out["tune"]["change"] < 0 and out["holdout"]["change"] < -out["holdout"]["se"]
    return out


def run_total(extra, feats):
    base_t = list(model.TOTAL_FEATURES)
    try:
        model.TOTAL_FEATURES[:] = base_t + feats
        return run(extra)
    finally:
        model.TOTAL_FEATURES[:] = base_t


if __name__ == "__main__":
    g = data.all_games(2026)
    sf = schedule_features(g)
    base = run()
    res = {}
    for name, feats in (("travel_km", ["km_diff"]), ("time_zones", ["tz_diff"]), ("altitude", ["alt"]), ("three_in_four", ["b3in4_diff"])):
        res[name] = r = compare_col(base, run(sf[["game_id"] + feats], feats), "m_margin")
        print(name, json.dumps(r), flush=True)
    # totals: pace gap and travel
    sf2 = sf[["game_id", "km_h", "km_a", "b3in4_h", "b3in4_a"]].assign(km_sum=lambda d: d.km_h + d.km_a, b3in4_sum=lambda d: d.b3in4_h + d.b3in4_a)
    for name, feats in (("total_travel", ["km_sum"]), ("total_three_in_four", ["b3in4_sum"])):
        res[name] = r = compare_col(base, run_total(sf2[["game_id"] + feats], feats), "m_total")
        print(name, json.dumps(r), flush=True)
    OUT.write_text(json.dumps(res, indent=1))
