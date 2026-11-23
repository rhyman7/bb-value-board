"""Sharp-price flags: Pinnacle's no-vig price sets the fair chance; US books that beat it are flagged."""
import json

import pandas as pd

from nbamodel import sharp


def _event(books):
    return {"home_team": "Boston Celtics", "away_team": "New York Knicks", "bookmakers": [
        {"key": k, "markets": [{"key": "spreads", "outcomes": [
            {"name": "Boston Celtics", "point": hp, "price": hpr}, {"name": "New York Knicks", "point": -hp, "price": apr}]}]}
        for k, hp, hpr, apr in books]}


GAMES = {("Boston Celtics", "New York Knicks"): ("g1", 60)}


def test_no_flag_when_books_match_pinnacle():
    js = [_event([("pinnacle", -5.5, -105, -105), ("draftkings", -5.5, -110, -110)])]
    assert sharp.flags(js, GAMES, "t") == []


def test_extra_points_flagged_on_right_side():
    # Pinnacle: Boston -5.5 at a fair 50/50. DraftKings has Knicks +8.5 at -110: about 59% to cover.
    js = [_event([("pinnacle", -5.5, -105, -105), ("draftkings", -8.5, -110, -110)])]
    rows = sharp.flags(js, GAMES, "t")
    assert len(rows) == 1
    r = rows[0]
    assert r["side"] == "away" and r["team"] == "New York Knicks" and r["spread"] == 8.5
    assert 0.58 < r["fair"] < 0.61 and r["ev"] > 0.1


def test_game_without_pinnacle_or_not_tonight_skipped():
    js = [_event([("draftkings", -8.5, -110, -110)])]
    assert sharp.flags(js, GAMES, "t") == []
    js = [_event([("pinnacle", -5.5, -105, -105), ("draftkings", -8.5, -110, -110)])]
    assert sharp.flags(js, {}, "t") == []


def test_consensus_sign_and_count():
    js = [_event([("pinnacle", -5.5, -105, -105), ("draftkings", -6.5, -110, -110), ("fanduel", -6.0, -110, -110)])]
    assert sharp.consensus(js) == {("Boston Celtics", "New York Knicks"): (6.0, 3)}


def test_grade_record_and_clv(tmp_path, monkeypatch):
    monkeypatch.setattr(sharp, "LOG", tmp_path / "s.csv")
    monkeypatch.setattr(sharp, "OUT", tmp_path / "s.json")
    base = {"home": "H", "away": "A", "minutes_to_tip": 60, "book": "DraftKings", "pin_spread": -5.5,
            "pin_home_price": -105, "pin_away_price": -105, "fair": 0.59}
    rows = [dict(base, fetched_at="2026-11-20T23:00:00+00:00", game_id="g1", side="away", team="A", spread=8.5, price=-110, ev=0.12),
            dict(base, fetched_at="2026-11-20T23:30:00+00:00", game_id="g1", side="away", team="A", spread=7.5, price=-110, ev=0.05),
            dict(base, fetched_at="2026-11-20T23:00:00+00:00", game_id="g2", side="home", team="H", spread=3.0, price=100, ev=0.04)]
    sharp.log(rows)
    sharp.update_tonight(rows[1:], "2026-11-20T23:30:00+00:00")
    from nbamodel import data
    # g1: home won by 8, away +8.5 covers. g2: home lost by 4, home +3 loses.
    monkeypatch.setattr(data, "games", lambda s: pd.DataFrame({"game_id": ["g1", "g2"], "margin": [8, -4]}))
    pre = pd.DataFrame({"game_id": ["g1", "g2"], "fetched_at": ["x", "x"], "line": [6.0, -1.0]})   # + = home favored
    out = sharp.grade(2027, pre)
    rc = out["record"]
    assert (rc["w"], rc["l"], rc["p"]) == (1, 1, 0)
    assert abs(rc["units"] - (100 / 110 - 1)) < 0.01
    # g1 away +8.5 vs closing away +6: +2.5. g2 home +3 vs closing home +1: +2.
    assert rc["avg_clv_pts"] == 2.25 and rc["beat_close_pct"] == 100.0
    assert [r["game_id"] for r in json.loads((tmp_path / "s.json").read_text())["tonight"]] == ["g1", "g2"]
