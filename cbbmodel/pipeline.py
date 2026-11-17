"""Tonight's college projections: rate every Division I team as of today and project the next
three days' games between two Division I teams. Writes docs/feed/cbb.json.

    python -m cbbmodel.pipeline

The board shows the model's margin next to the market's spread. The backtest has not earned the
model a weight against the line (see README), so `projection` is the model's own number.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nbamodel.ratings import rating_projection

from . import backtest, data, ratings, roster

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "docs" / "feed" / "cbb.json"
DAYS_AHEAD = 3


def run():
    today = datetime.now(ET).date()
    season = today.year + 1 if today.month >= 8 else today.year
    g = data.all_games(season)
    g_done = g[g.date < pd.Timestamp(today)]
    shifts = roster.prior_shift(range(backtest.FIRST_RATED, season + 1))
    seasons = list(range(backtest.FIRST_RATED, season + 1))
    rated, final = ratings.walk_forward(g_done, backtest.SETTINGS, seasons=seasons, shifts=shifts, return_final=True)
    rated = backtest.add_features(rating_projection(rated))
    coef = backtest.fit(rated[rated.season < season])

    sched = data.schedule(season, refresh=False)
    d1 = data.division_one(sched)
    s = sched[(~sched.status_type_completed.astype(bool)) & sched.home_id.astype(int).isin(d1) & sched.away_id.astype(int).isin(d1)].copy()
    s["et"] = pd.to_datetime(s.date, utc=True).dt.tz_convert(ET)
    s["day"] = s.et.dt.date
    s = s[(s.day >= today) & (s.day < today + timedelta(days=DAYS_AHEAD))]
    n_this = int((g_done.season == season).sum())
    last_played = {}
    for x in g_done[g_done.season == season].itertuples():
        for t in (x.home_id, x.away_id):
            last_played[t] = max(last_played.get(t, x.date), x.date)

    games = []
    for x in s.sort_values("et").itertuples():
        hid, aid = int(x.home_id), int(x.away_id)
        rh, ra = final.get(hid, (0.0, 0.0, 0.0)), final.get(aid, (0.0, 0.0, 0.0))
        day = pd.Timestamp(x.day)
        rest = lambda t: int(min(4, max(0, (day - last_played[t]).days - 1))) if t in last_played else 4
        row = pd.DataFrame([{
            "off_h": rh[0], "def_h": rh[1], "pace_h": rh[2], "off_a": ra[0], "def_a": ra[1], "pace_a": ra[2],
            "lg_eff": final["_mean"], "lg_hca": backtest.SETTINGS.hca, "lg_pace": final["_pace"],
            "neutral": bool(x.neutral_site), "rest_h": rest(hid), "rest_a": rest(aid), "g7_h": 0, "g7_a": 0, "n_this": n_this,
        }])
        row = backtest.predict(backtest.add_features(rating_projection(row)), coef).iloc[0]
        m = float(row.m_margin)
        games.append({
            "id": str(x.game_id), "date": x.day.isoformat(), "away": x.away_abbreviation, "home": x.home_abbreviation,
            "neutral": bool(x.neutral_site), "margin": round(m, 1), "total": round(float(row.r_total), 1),
            "pace": round(float(row.r_pace), 1), "win_prob": round(float(1 / (1 + np.exp(-m / 6.5))), 3),
            "projection": round(m, 1),
        })
    feed = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "gamesRated": n_this, "marketWeight": 0.0,
        "coefficients": {k: round(v, 3) for k, v in coef.items()},
        "games": games,
    }
    FEED.parent.mkdir(parents=True, exist_ok=True)
    FEED.write_text(json.dumps(feed, indent=1))
    print(f"{len(games)} college games projected for {today} to {today + timedelta(days=DAYS_AHEAD - 1)}")
    return feed


if __name__ == "__main__":
    run()
