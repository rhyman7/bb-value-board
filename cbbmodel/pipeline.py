"""Tonight's college projections: rate every Division I team as of today and project the next
three days' games between two Division I teams. Writes docs/feed/cbb.json, and logs ESPN's line
with the model's number for each game (docs/lines/cbb_lines.csv, see nbamodel/lines.py).

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

from nbamodel import lines as line_log
from nbamodel.ratings import rating_projection

from . import backtest, data, ratings, roster, spots

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "docs" / "feed" / "cbb.json"
DAYS_AHEAD = 3


SPOT_TEXT = {"letdown": "{t} off an emotional win (letdown spot)", "lookahead": "{t} has a key game next (lookahead spot)",
             "road_b2b_tz": "{t} on a second straight road game, new time zone", "road_b2b": "{t} on a second straight road game in 4 days",
             "far_road": "{t} is 2+ time zones from home"}


def upcoming_spots(g_done, rated, sched, d1, final, today, lines):
    """Situational spots for every game left on the schedule, as short labels per game id. They are
    shown for context; only the long road trip (far_road) is in the model, as a small adjustment."""
    s = sched[(~sched.status_type_completed.astype(bool)) & sched.home_id.astype(int).isin(d1) & sched.away_id.astype(int).isin(d1)].copy()
    s["date"] = pd.to_datetime(pd.to_datetime(s.date, utc=True).dt.tz_convert(ET).dt.date)
    s = s[s.date >= pd.Timestamp(today)]
    if s.empty:
        return {}, {}
    fut = pd.DataFrame({"game_id": s.game_id.astype(str), "season": s.season.astype(int), "date": s.date,
                        "home_id": s.home_id.astype(int), "away_id": s.away_id.astype(int),
                        "home": s.home_abbreviation, "away": s.away_abbreviation,
                        "neutral": s.neutral_site.fillna(False).astype(bool), "venue_address_state": s.venue_address_state})
    for side, col in (("h", "home_id"), ("a", "away_id")):
        r = fut[col].map(lambda t: final.get(t, (0.0, 0.0, 0.0)))
        fut[f"off_{side}"], fut[f"def_{side}"], fut[f"pace_{side}"] = [r.map(lambda v, i=i: v[i]) for i in range(3)]
    fut["lg_eff"], fut["lg_hca"], fut["lg_pace"] = final["_mean"], backtest.SETTINGS.hca, final["_pace"]
    fut = rating_projection(fut)
    fut["close_spread"] = fut.game_id.map(lambda i: lines.get(i, (None, None))[0]).astype(float)
    fut["margin"] = np.nan
    cols = ["game_id", "season", "date", "home_id", "away_id", "neutral", "venue_address_state", "close_spread",
            "r_margin", "margin", "off_h", "def_h", "off_a", "def_a"]
    season = int(fut.season.max())
    past = rated[rated.season >= season - 11]
    both = pd.concat([past[cols], fut[cols]], ignore_index=True)
    sp = spots.spots(pd.concat([g_done[g_done.season >= season - 11], fut], ignore_index=True), both)
    sp = sp[sp.game_id.isin(set(fut.game_id))].merge(fut[["game_id", "home", "away"]], on="game_id")
    out, far = {}, {}
    for x in sp.itertuples():
        far[x.game_id] = float(x.far_road_diff)
        tags = []
        for side, team in (("h", x.home), ("a", x.away)):
            for k in ["letdown", "lookahead", "road_b2b_tz", "road_b2b", "far_road"]:
                if getattr(x, f"{k}_{side}") > 0 and not (k == "road_b2b" and getattr(x, f"road_b2b_tz_{side}") > 0):
                    tags.append(SPOT_TEXT[k].format(t=team))
        if x.rival_home_dog > 0:
            tags.append(f"Rivalry game, {x.home} a home underdog")
        elif x.rival:
            tags.append("Rivalry game")
        out[x.game_id] = tags
    return out, far


def run():
    today = datetime.now(ET).date()
    season = today.year + 1 if today.month >= 8 else today.year
    g = data.all_games(season)
    g_done = g[g.date < pd.Timestamp(today)]
    shifts = roster.prior_shift(range(backtest.FIRST_RATED, season + 1))
    seasons = list(range(backtest.FIRST_RATED, season + 1))
    rated, final = ratings.walk_forward(g_done, backtest.SETTINGS, seasons=seasons, shifts=shifts, return_final=True)
    rated = backtest.add_features(rating_projection(rated))
    rated = rated.merge(spots.spots(g_done, rated), on="game_id", how="left")
    rated["far_road_diff"] = rated.far_road_diff.fillna(0.0)
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

    lines = line_log.espn_lines("cbb", sorted(set(s.day)))
    tags, far = upcoming_spots(g_done, rated, sched, d1, final, today, lines)
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
            "far_road_diff": far.get(str(x.game_id), 0.0),
        }])
        row = backtest.predict(backtest.add_features(rating_projection(row)), coef).iloc[0]
        m = float(row.m_margin)
        sp, tot = lines.get(str(x.game_id), (None, None))
        games.append({
            "id": str(x.game_id), "date": x.day.isoformat(), "away": x.away_abbreviation, "home": x.home_abbreviation,
            "neutral": bool(x.neutral_site), "margin": round(m, 1), "total": round(float(row.r_total), 1),
            "pace": round(float(row.r_pace), 1), "win_prob": round(float(1 / (1 + np.exp(-m / 6.5))), 3),
            "projection": round(m, 1), "mkt_spread": sp, "mkt_total": tot,
            "spots": tags.get(str(x.game_id), []),
        })
    feed = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "gamesRated": n_this, "marketWeight": 0.0,
        "coefficients": {k: round(v, 3) for k, v in coef.items()},
        "games": games,
    }
    FEED.parent.mkdir(parents=True, exist_ok=True)
    FEED.write_text(json.dumps(feed, indent=1))
    n_lines = line_log.log("cbb", games)
    print(f"{len(games)} college games projected for {today} to {today + timedelta(days=DAYS_AHEAD - 1)}; {n_lines} with an ESPN line")
    return feed


if __name__ == "__main__":
    run()
