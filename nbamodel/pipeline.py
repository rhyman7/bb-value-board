"""Tonight's projections: rate every team as of today, project the next few days' games, and write
the feed the NBA dashboard reads (docs/feed/nba.json) plus a readable board (docs/nba/index.html).

    python -m nbamodel.pipeline [--backtest]

Missing players come from ESPN's injury report (players listed Out), valued the same way as in the
backtest. When the report can't be read the projections assume full rosters and say so.
The market blend uses the weight the backtest has earned on the closing line (currently none, so the
projection shown is the model's own number next to the market's).
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import archive, availability, backtest, data, gamepage, impact, lines as line_log, market, model, players, ratings, soft

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
FEED = ROOT / "docs" / "feed" / "nba.json"
DAYS_AHEAD = 3
TEAM_NAME = {"LA Clippers": "Los Angeles Clippers"}   # ESPN name -> the Odds API's, where they differ
ESPN_INJURIES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"


def _json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "bb-value-board"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def injuries():
    """{athlete_id: status} for players ESPN lists Out (or Doubtful). None when unreachable."""
    try:
        js = _json(ESPN_INJURIES)
    except Exception:
        return None
    out = {}
    for team in js.get("injuries", []):
        for it in team.get("injuries", []):
            st = (it.get("status") or "").lower()
            ath = it.get("athlete") or {}
            aid = ath.get("id") or (ath.get("links") and None)
            if aid and st in ("out", "doubtful"):
                out[int(aid)] = st
    return out


def live_market(g, season, day):
    """Market ratings for today from the closes of earlier games: hoopR's closing lines where it
    has them, else the last ESPN line the pipeline logged. Returns solve()'s output or None."""
    hist = g[g.season.isin([season - 1, season]) & (g.date < day)]
    closes = data.closing_lines()
    known = hist.merge(closes[["game_id", "close_spread", "close_total"]], on="game_id", how="left")
    logged = line_log.last_lines("nba").rename(columns={"close_spread": "ls", "close_total": "lt"})
    known = known.merge(logged, on="game_id", how="left")
    known["close_spread"] = known.close_spread.fillna(known.ls)
    known["close_total"] = known.close_total.fillna(known.lt)
    try:
        known = known.merge(availability.missing_value(g, [season - 1, season])[["game_id", "miss_h", "miss_a"]], on="game_id", how="left")
    except Exception:
        known["miss_h"] = known["miss_a"] = 0.0
    known = known.dropna(subset=["close_spread", "close_total"])
    if (known.season == season).sum() < market.MIN_THIS_SEASON:
        return None
    return market.solve(known, day, season)


def upcoming(season, today):
    s = data.schedule(season, refresh=True)
    s = s[(s.season_type == 2) & (~s.status_type_completed.astype(bool))].copy()
    s["et"] = pd.to_datetime(s.date, utc=True).dt.tz_convert(ET)
    s["day"] = s.et.dt.date
    s = s[(s.day >= today) & (s.day < today + timedelta(days=DAYS_AHEAD)) & (s.home_abbreviation != "TBD")]
    return s


def current_missing(g, season, inj):
    """Missing value per team id for today, from the injury report, valued like the backtest."""
    if inj is None:
        return {}, {}, {}
    pb = availability.player_box(season, refresh=True)
    played = pb[pb.minutes > 0].sort_values("date")
    try:
        prev = availability.player_box(season - 1)
        prev_tot = prev[prev.minutes > 0].groupby("athlete_id")[["gmsc", "minutes"]].sum()
    except Exception:
        prev_tot = pd.DataFrame(columns=["gmsc", "minutes"])
    tot = played.groupby("athlete_id")[["gmsc", "minutes"]].sum()
    last_team = played.groupby("athlete_id").team_id.last()
    miss, names, detail = {}, {}, {}
    for tid, tp in played.groupby("team_id"):
        recent_games = tp.game_id.drop_duplicates().tail(availability.WINDOW)
        r = tp[tp.game_id.isin(recent_games)]
        n_games = len(recent_games)
        for aid, pr in r.groupby("athlete_id"):
            if aid not in inj or last_team.get(aid) != tid:
                continue
            share = len(pr) / max(n_games, 1)
            role = pr.minutes.mean()
            if share < 0.3 or role < 10:
                continue
            g_, m_ = tot.loc[aid] if aid in tot.index else (0.0, 0.0)
            pg, pm = prev_tot.loc[aid] if aid in prev_tot.index else (0.0, 0.0)
            rate = (g_ + 0.5 * pg + availability.REPLACEMENT * availability.PRIOR_MIN) / (m_ + 0.5 * pm + availability.PRIOR_MIN)
            v = max(0.0, rate - availability.REPLACEMENT) * role
            miss[tid] = miss.get(tid, 0.0) + v
            if v > 1:
                names.setdefault(tid, []).append(pr.athlete_display_name.iloc[-1])
            if v > 0.3:
                detail.setdefault(tid, []).append((pr.athlete_display_name.iloc[-1], float(v)))
    return miss, names, detail


def run(do_backtest: bool = False):
    now = datetime.now(ET)
    today = now.date()
    season = today.year + 1 if today.month >= 8 else today.year
    g = data.all_games(season)
    s = ratings.Settings()

    # coefficients: fitted on every finished season (the backtest refits season by season)
    hist_seasons = [x for x in sorted(g.season.unique()) if x > g.season.min() and x < season]
    rated = ratings.rating_projection(ratings.walk_forward(g, s, seasons=hist_seasons))
    rated = rated.merge(availability.missing_value(g, hist_seasons), on="game_id", how="left")
    rated = rated.merge(impact.history(hist_seasons)[["game_id", "imp_diff"]], on="game_id", how="left")
    rated = rated.merge(data.closing_lines(), on="game_id", how="left")
    rated = rated.merge(market.market_ratings(rated, [x for x in hist_seasons if x >= backtest.GRADE_FROM]), on="game_id", how="left")
    coef = model.fit(model.features(rated))

    bt_summary = None
    bt_path = ROOT / "docs" / "nba" / "backtest.json"
    if do_backtest:
        bt_res = backtest.run(season, s, verbose=False)
        bt_summary = bt_res["summary"]
        try:   # refit the first-half scaling on the same walk-forward games
            soft.fit(bt_res["games"][bt_res["games"].season >= 2021])
        except Exception as e:
            print("first-half fit failed:", repr(e))
        bt_path.parent.mkdir(parents=True, exist_ok=True)
        bt_path.write_text(json.dumps(bt_summary, indent=1))
    elif bt_path.exists():
        bt_summary = json.loads(bt_path.read_text())
    w_side = max(0.0, (bt_summary or {}).get("market_weight", {}).get("side", 0.0) or 0.0)
    w_total = max(0.0, min(0.3, (bt_summary or {}).get("market_weight", {}).get("total", 0.0) or 0.0))

    # today's ratings
    day = pd.Timestamp(today)
    pool = g[g.season.isin([season - 1, season]) & (g.date < day)]
    R, mean, hca, pmean = ratings.rate_day(pool, day, season, s)

    sched = upcoming(season, today)
    inj = injuries()
    miss, miss_names, miss_detail = current_missing(g, season, inj)
    try:   # impact ratings as of today and each team's lineup strength tonight (players listed out left off)
        imp_str, imp_rating, imp_names = impact.today(season, set(inj or {}))
    except Exception as e:
        print("impact ratings failed:", repr(e))
        imp_str, imp_rating, imp_names = {}, {}, {}
    lines = line_log.espn_lines("nba", sorted(set(sched.day)))
    mk = live_market(g, season, day)
    this = g[g.season == season]
    last_played = {}
    for x in this.itertuples():
        last_played[x.home_id] = max(last_played.get(x.home_id, x.date), x.date)
        last_played[x.away_id] = max(last_played.get(x.away_id, x.date), x.date)

    games = []
    for x in sched.sort_values("et").itertuples():
        hid, aid = int(x.home_id), int(x.away_id)
        gday = pd.Timestamp(x.day)
        # rest: days off since each team's last game (scheduled games in between count)
        def rest(tid):
            prev = [d for d in [last_played.get(tid)] if d is not None]
            earlier = sched[((sched.home_id.astype(int) == tid) | (sched.away_id.astype(int) == tid)) & (sched.day < x.day)]
            prev += [pd.Timestamp(d) for d in earlier.day]
            if not prev:
                return 4
            return int(min(4, max(0, (gday - max(prev)).days - 1)))
        def g7(tid):
            ds = [d for d in this[(this.home_id == tid) | (this.away_id == tid)].date if 0 < (gday - d).days <= 6]
            return len(ds)
        rh, ra = R.get(hid, (0, 0, 0)), R.get(aid, (0, 0, 0))
        row = pd.DataFrame([{
            "off_h": rh[0], "def_h": rh[1], "pace_h": rh[2], "off_a": ra[0], "def_a": ra[1], "pace_a": ra[2],
            "lg_eff": mean, "lg_hca": hca, "lg_pace": pmean, "neutral": bool(x.neutral_site),
            "rest_h": rest(hid), "rest_a": rest(aid), "g7_h": g7(hid), "g7_a": g7(aid),
            "miss_h": miss.get(hid, 0.0), "miss_a": miss.get(aid, 0.0), "n_this": len(this),
            "imp_h": imp_str.get(hid), "imp_a": imp_str.get(aid),
            "imp_diff": imp_str[hid] - imp_str[aid] if hid in imp_str and aid in imp_str else np.nan,
            "mk_margin": (mk[0][hid] - mk[0][aid] + (0.0 if x.neutral_site else mk[1])) if mk and hid in mk[0] and aid in mk[0] else np.nan,
            "mk_total": (mk[2] + mk[3][hid] + mk[3][aid]) if mk and hid in mk[3] and aid in mk[3] else np.nan,
        }])
        row = model.predict(model.features(ratings.rating_projection(row)), coef).iloc[0]
        gid = str(x.game_id)
        build = gamepage.build(row, coef, x.home_abbreviation, x.away_abbreviation,
                               {"h": miss_detail.get(hid, []), "a": miss_detail.get(aid, [])}, bool(row.has_mk))
        mkt_spread, mkt_total, ml_h, ml_a = lines.get(gid, (None, None, None, None))
        proj = float(row.m_margin) if mkt_spread is None else mkt_spread + w_side * (float(row.m_margin) - mkt_spread)
        projt = float(row.m_total) if mkt_total is None else mkt_total + w_total * (float(row.m_total) - mkt_total)
        games.append({
            "id": gid, "date": x.day.isoformat(), "tip": x.et.isoformat(), "away": x.away_abbreviation, "home": x.home_abbreviation,
            "margin": round(float(row.m_margin), 1), "total": round(float(row.m_total), 1),
            "ratings_margin": round(float(row.r_margin), 1), "pace": round(float(row.r_pace), 1),
            "market_ratings_margin": round(float(row.mk_margin), 1) if row.has_mk else None,
            "mkt_spread": mkt_spread, "mkt_total": mkt_total, "mkt_ml_home": ml_h, "mkt_ml_away": ml_a,
            "projection": round(proj, 1), "projection_total": round(projt, 1),
            "win_prob": round(float(1 / (1 + np.exp(-float(row.m_margin) / 7.0))), 3),
            "rest": {"away": int(row.rest_a), "home": int(row.rest_h)},
            "build": build,
            "imp": {"home": None if row.imp_h is None or pd.isna(row.imp_h) else round(float(row.imp_h), 3),
                    "away": None if row.imp_a is None or pd.isna(row.imp_a) else round(float(row.imp_a), 3)},
            "missing": {"away": round(float(row.miss_a), 1), "home": round(float(row.miss_h), 1),
                        "away_names": miss_names.get(aid, []), "home_names": miss_names.get(hid, [])},
        })

    teams = {}
    abbr = {}
    for x in this.itertuples():
        abbr[x.home_id] = x.home
        abbr[x.away_id] = x.away
    for tid, (o, d, p) in R.items():
        if tid in abbr:
            teams[abbr[tid]] = {"off": round(mean + o, 1), "def": round(mean - d, 1), "net": round(o + d, 1), "pace": round(pmean + p, 1)}

    # game pages: four factors with ranks and each team's last ten games
    try:
        tstats = gamepage.team_stats(data.team_box(season), {int(k): v for k, v in abbr.items()}, data.possessions)
    except Exception:
        tstats = {}
    try:
        esp = data.espn_spreads(season, refresh=True)
    except Exception:
        esp = None
    form = gamepage.last_ten(this, esp)
    for ab, t in teams.items():
        t["stats"] = tstats.get(ab)
        t["last10"] = form.get(ab, [])

    # player projections for every game on the board; today's are logged and graded the next morning
    n_players, pgrade = 0, None
    try:
        if do_backtest or not players.NBA.params_file.exists():
            players.run()
        pst = players.state_today(season)
        pid = {int(k): v for k, v in abbr.items()}
        proj = players.tonight(pst, [(x["id"], int(r.home_id), int(r.away_id)) for x, r in
                                     zip(games, sched.sort_values("et").itertuples())], set(inj or {}), pid)
        todays = {k: v for k, v in proj.items() if any(x["id"] == k and x["date"] == today.isoformat() for x in games)}
        # prop lines for today's games, once a day on the late-afternoon run (about 4 credits a game)
        if os.environ.get("ODDS_API_KEY") and now.hour >= 16 and todays:
            try:
                from oddsapi import props as oprops
                full = {}
                for r in sched.itertuples():
                    if str(r.game_id) in todays:
                        full[str(r.game_id)] = (TEAM_NAME.get(r.home_display_name, r.home_display_name), TEAM_NAME.get(r.away_display_name, r.away_display_name))
                n_props = oprops.attach(todays, oprops.today_props(full, today.isoformat()))
                print(f"prop lines attached for {n_props} players")
            except Exception as e:
                print("props failed:", repr(e))
        for x in games:
            x["players"] = proj.get(x["id"])
        n_players = players.log(today, todays)
        pgrade = players.grade_log(season)
    except Exception as e:  # the board still goes out without them
        print("player projections failed:", repr(e))

    feed = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "injuryReport": inj is not None,
        "playerGrades": pgrade,
        "marketRatings": mk is not None,
        "marketWeight": {"side": round(w_side, 3), "total": round(w_total, 3)},
        "homeCourt": round(hca * pmean / 100, 2),
        "coefficients": {k: {a: round(b, 3) for a, b in v.items()} for k, v in coef.items()},
        "games": games,
        "teams": teams,
    }
    FEED.parent.mkdir(parents=True, exist_ok=True)
    FEED.write_text(json.dumps(feed, indent=1, allow_nan=False))
    line_log.log("nba", games)
    # first-half lines read with the props, and the morning grade of the ones that have finished
    try:
        from oddsapi import props as oprops
        if oprops.GAME_LINES:
            print(f"first-half lines logged for {soft.log(today, games, oprops.GAME_LINES)} games")
        if soft.LOG.exists():
            soft.grade(season)
    except Exception as e:
        print("first-half log failed:", repr(e))
    # past nights: freeze today's games before tip-off, grade the ones that have finished
    archive.freeze("nba", games, now)
    archive.grade("nba", g[g.season == season], f"{season - 1}-10-01")
    try:   # the pre-tip injury-news test (pretip.py)
        from . import pretip
        pretip.grade(season)
    except Exception as e:
        print("pre-tip grade failed:", repr(e))
    print(f"{len(games)} games projected for {today} to {today + timedelta(days=DAYS_AHEAD - 1)}; "
          f"injury report {'read' if inj is not None else 'NOT read (full rosters assumed)'}; "
          f"{sum(1 for x in games if x['mkt_spread'] is not None)} with a market line; {n_players} player projections logged")
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true", help="rerun the full backtest first")
    args = ap.parse_args()
    run(args.backtest)


if __name__ == "__main__":
    main()
