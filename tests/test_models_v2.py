import numpy as np
import pandas as pd

from cbbmodel import ratings as cbb_ratings
from nbamodel import market


def _games(season=2027, n_rounds=6):
    rows = []
    day = pd.Timestamp("2026-11-01")
    pairs = [("A", "B", 80, 65), ("B", "C", 75, 70), ("C", "A", 60, 82)] * n_rounds
    for i, (h, a, ph, pa) in enumerate(pairs):
        rows.append({"game_id": f"{season}-{i}", "season": season, "date": day + pd.Timedelta(days=i), "home_id": h, "away_id": a,
                     "pts_h": ph, "pts_a": pa, "poss": 68.0, "neutral": False})
    g = pd.DataFrame(rows)
    g["margin"] = g.pts_h - g.pts_a
    return g


def test_college_ratings_order_teams_and_carry_prior():
    g = pd.concat([_games(2026, 15), _games(2027, 12)], ignore_index=True)
    r, final = cbb_ratings.walk_forward(g, cbb_ratings.Settings(), seasons=[2027], return_final=True)
    net = {t: final[t][0] + final[t][1] for t in "ABC"}
    assert net["A"] > net["B"] > net["C"]
    # the first game of 2027 already knows A is good, from last season
    first = r.sort_values("date").iloc[0]
    assert (first.off_h + first.def_h) > (first.off_a + first.def_a)


def test_market_ratings_recover_spreads():
    g = _games(2027, 80)
    g["close_spread"] = np.where(g.home_id == "A", 12.0, np.where(g.home_id == "B", 4.0, -10.0))
    g["close_total"] = 140.0
    g["miss_h"] = g["miss_a"] = 0.0
    R, hca, mu, T = market.solve(g, g.date.max() + pd.Timedelta(days=1), 2027)
    assert R["A"] > R["B"] > R["C"]
    assert abs(mu - 140.0) < 1e-6


def test_openers_grades_line_movement(tmp_path, monkeypatch):
    from nbamodel import lines, openers
    log = tmp_path / "nba_lines.csv"
    rows = []
    for i in range(10):
        # model says home by 6, opener home by 2, close moves to home by 4; home wins by 8
        rows.append({"game_id": str(i), "fetched_at": "2026-11-17T10:00:00+00:00", "close_spread": 2.0, "close_total": 220.0, "model_margin": 6.0, "model_total": 225.0})
        rows.append({"game_id": str(i), "fetched_at": "2026-11-17T22:00:00+00:00", "close_spread": 4.0, "close_total": 222.0, "model_margin": 6.0, "model_total": 225.0})
    pd.DataFrame(rows).to_csv(log, index=False)
    monkeypatch.setitem(lines.LOG, "nba", log)
    monkeypatch.setattr(openers, "_results", lambda league, season: pd.DataFrame({"game_id": [str(i) for i in range(10)], "margin": 8.0, "total": 230.0}))
    out = openers.grade("nba", 2027)
    g = out["side"]["by_gap"]["3+"]
    assert g["record_at_open"]["w"] == 10 and g["line_moved_toward_model_pct"] == 100.0
    assert out["total"]["by_gap"]["5+"]["record_at_open"]["w"] == 10


def test_moneyline_parse_and_grade(tmp_path, monkeypatch):
    from nbamodel import lines, openers
    assert lines._ml({"moneyLine": -150}, {}, "home") == -150
    assert lines._ml(None, {"moneyline": {"away": {"close": {"odds": "+130"}}}}, "away") == 130
    assert lines._ml(None, {}, "home") is None
    log = tmp_path / "nba_lines.csv"
    # an old log without moneyline columns gets them added on the next write
    pd.DataFrame([{"game_id": "x", "fetched_at": "2026-11-16T10:00:00+00:00", "close_spread": 1.0, "close_total": 220.0,
                   "model_margin": 2.0, "model_total": 221.0}]).to_csv(log, index=False)
    monkeypatch.setitem(lines.LOG, "nba", log)
    games = [{"id": str(i), "mkt_spread": -3.0, "mkt_total": 220.0, "margin": 1.0, "total": 221.0,
              "mkt_ml_home": 140, "mkt_ml_away": -160, "win_prob": 0.55} for i in range(10)]
    assert lines.log("nba", games) == 10
    assert list(pd.read_csv(log).columns) == lines.COLUMNS
    # home dog at +140 (no-vig ~40%), model says 55%: bet home every game; home wins 6 of 10
    monkeypatch.setattr(openers, "_results", lambda league, season: pd.DataFrame(
        {"game_id": [str(i) for i in range(10)], "margin": [5.0] * 6 + [-5.0] * 4, "total": 220.0}))
    ml = openers.grade("nba", 2027)["moneyline"]
    e = ml["by_edge"]["4%+"]
    assert ml["games"] == 10 and e["bets"] == 10 and e["won"] == 6
    assert abs(e["roi_pct"] - 100 * (6 * 1.4 - 4) / 10) < 0.1


def test_player_projection_shape():
    """team_rows/project on a tiny made-up season: projections exist, are non-negative, minutes
    capped, and a missing regular shows up in the out list."""
    import pandas as pd
    from nbamodel import players as P
    st = P.State()
    rows = []
    for g in range(6):
        for tid, opp in ((1, 2), (2, 1)):
            for a in range(8):
                aid = tid * 100 + a
                rows.append({"game_id": f"g{g}", "team_id": tid, "opponent_team_id": opp, "athlete_id": aid,
                             "athlete_display_name": f"P{aid}", "minutes": 36 - 3 * a, "points": 20 - 2 * a,
                             "rebounds": 5, "assists": 3, "threes": 1})
    df = pd.DataFrame(rows)
    for _, gp in df.groupby("game_id"):
        st.add_game(gp)
    r = P.team_rows(st, 1, 2, {100})
    assert r and all(x["athlete_id"] != 100 for x in r)
    assert r[0]["out"] == ["P100"]
    params = {"minutes": {"const": 0.0, "ewm15": 0.0, "ewm30": 1.0, "role": 0.0, "season_avg": 0.0, "last_game": 0.0, "teammates_out": 1.0},
              "beta": {s: 1.0 for s in P.STATS}, "u": {s: 0.5 for s in P.STATS}}
    p = P.project(pd.DataFrame(r), params)
    assert (p.p_minutes <= P.MAX_MIN).all() and (p[[f"p_{s}" for s in P.STATS]] >= 0).all().all()
    assert p.p_minutes.iloc[0] > 33   # picks up some of the missing starter's minutes
