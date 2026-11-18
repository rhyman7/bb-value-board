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

from nbamodel import archive, gamepage, lines as line_log, players
from nbamodel.ratings import rating_projection

from . import availability, backtest, data, players as cbb_players, ratings, roster, spots

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "docs" / "feed" / "cbb.json"
PFEED = ROOT / "docs" / "feed" / "cbb_players.json"
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
    rated = rated.merge(availability.missing_value(g_done, seasons)[["game_id", "miss_h", "miss_a"]], on="game_id", how="left")
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
    # who is out: no college injury report, so regulars who sat their last 2 games, plus anyone ESPN
    # lists Out or Doubtful for today's games (ESPN's college coverage is thin)
    espn_out, espn_note = availability.espn_injuries([str(x) for x in s[s.day == today].game_id])
    try:
        miss, miss_detail, out_ids = availability.current(season, espn_out)
    except Exception as e:
        print("missing players failed:", repr(e))
        miss, miss_detail, out_ids = {}, {}, set()
    print(espn_note, f"; {len(out_ids)} regulars counted out")
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
            "miss_h": miss.get(hid, 0.0), "miss_a": miss.get(aid, 0.0),
        }])
        row = backtest.predict(backtest.add_features(rating_projection(row)), coef).iloc[0]
        m = float(row.m_margin)
        fade = float(np.clip(1 - n_this / backtest.EARLY_GAMES, 0, 1))
        cm = {k: v for k, v in coef.items() if k != "r_early"}
        cm["r_margin"] = coef["r_margin"] + coef.get("r_early", 0.0) * fade
        build = gamepage.build(row, {"margin": cm, "total": {"const": 0.0, "r_total": 1.0}},
                               x.home_abbreviation, x.away_abbreviation,
                               {"h": miss_detail.get(hid, []), "a": miss_detail.get(aid, [])}, False)
        sp, tot, ml_h, ml_a = lines.get(str(x.game_id), (None, None, None, None))
        games.append({
            "id": str(x.game_id), "date": x.day.isoformat(), "tip": x.et.isoformat(), "away": x.away_abbreviation, "home": x.home_abbreviation,
            "neutral": bool(x.neutral_site), "margin": round(m, 1), "total": round(float(row.r_total), 1),
            "pace": round(float(row.r_pace), 1), "win_prob": round(float(1 / (1 + np.exp(-m / 6.5))), 3),
            "projection": round(m, 1), "mkt_spread": sp, "mkt_total": tot, "mkt_ml_home": ml_h, "mkt_ml_away": ml_a,
            "spots": tags.get(str(x.game_id), []),
            "missing": {"home": [n for n, _ in miss_detail.get(hid, [])], "away": [n for n, _ in miss_detail.get(aid, [])]},
            "build": build,
        })
    # game pages: four factors with ranks (among all of Division I) and last ten games, for the teams on the board
    playing = {x["home"] for x in games} | {x["away"] for x in games}
    abbr_of = {}
    for side in ("home", "away"):
        for tid, ab in zip(sched[f"{side}_id"].astype(int), sched[f"{side}_abbreviation"]):
            if tid in d1:
                abbr_of[tid] = ab
    try:
        tb = data.team_box(season)
        tb = tb[tb.team_id.astype(int).isin(d1) & tb.opponent_team_id.astype(int).isin(d1)]
        tstats = gamepage.team_stats(tb, abbr_of, data.possessions, only=playing)
    except Exception:
        tstats = {}
    this = g_done[g_done.season == season]
    form = gamepage.last_ten(this, this[["game_id"]].assign(espn_spread=this.close_spread).dropna())
    teams = {ab: {"net": round(float(final[t][0] + final[t][1]), 1) if t in final else None,
                  "stats": tstats.get(ab), "last10": form.get(ab, [])} for t, ab in abbr_of.items() if ab in playing}

    feed = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season, "gamesRated": n_this, "marketWeight": 0.0,
        "coefficients": {k: round(v, 3) for k, v in coef.items()},
        "games": games,
        "teams": teams,
        "nTeams": len(d1),
    }
    FEED.parent.mkdir(parents=True, exist_ok=True)
    FEED.write_text(json.dumps(feed, indent=1, allow_nan=False))

    # player projections: their own file (a college slate is too big to put in the main feed). No
    # injury report for college, so full rosters are assumed.
    n_players = 0
    try:
        L = cbb_players.CBB
        if not L.params_file.exists():
            players.run(L)
        pst = players.state_today(season, lg=L)
        sx = s.sort_values("et")
        proj = players.tonight(pst, [(x["id"], int(r.home_id), int(r.away_id)) for x, r in zip(games, sx.itertuples())], out_ids, abbr_of, lg=L)
        for x in games:
            if x["id"] in proj:
                proj[x["id"]]["date"] = x["date"]
        todays = {k: v for k, v in proj.items() if v.get("date") == today.isoformat()}
        n_players = players.log(today, todays, lg=L)
        PFEED.write_text(json.dumps({"generatedAt": feed["generatedAt"], "injuryReport": False, "inferredOut": len(out_ids),
                                     "games": proj, "playerGrades": players.grade_log(season, lg=L)}, allow_nan=False))
    except Exception as e:
        print("college player projections failed:", repr(e))
    n_lines = line_log.log("cbb", games)
    archive.freeze("cbb", games, datetime.now(ET))
    archive.grade("cbb", g[(g.season == season) & (g.date < pd.Timestamp(today))], f"{season - 1}-10-01")
    print(f"{len(games)} college games projected for {today} to {today + timedelta(days=DAYS_AHEAD - 1)}; {n_lines} with an ESPN line; {n_players} player projections logged")
    return feed


if __name__ == "__main__":
    run()
