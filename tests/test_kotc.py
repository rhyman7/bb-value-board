from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from nbamodel import kotc


def _game(gid, tip, players):
    return {"id": gid, "date": "2026-11-24", "tip": tip, "away": "A", "home": "H",
            "players": {"home": players, "away": [], "out": {"home": [], "away": []}}}


def _p(i, name, pts, reb, ast):
    return {"id": i, "name": name, "team": "H", "opp": "A", "points": pts, "rebounds": reb, "assists": ast,
            "avg": {"points": pts, "rebounds": reb, "assists": ast}}


def test_chances_sum_to_one_and_favor_the_bigger_projection():
    ch, med, tie = kotc.chances([45, 35, 35], ["1", "2", "2"], sims=20_000)
    assert abs(ch.sum() - 1) < 1e-9
    assert ch[0] > ch[1] > 0.1 and abs(ch[1] - ch[2]) < 0.03
    assert 40 < med < 60 and 0 <= tie < 0.2


def test_update_freezes_at_first_tip_and_grades(tmp_path, monkeypatch):
    monkeypatch.setattr(kotc, "OUT", tmp_path / "kotc.json")
    monkeypatch.setattr(kotc, "LOG", tmp_path / "kotc.csv")
    games = [_game("1", "2026-11-24T19:00:00-05:00", [_p(10, "Big", 28, 12, 9), _p(11, "Mid", 22, 5, 5), _p(12, "Low", 6, 3, 1)])]
    before = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)   # 3 PM ET
    d = kotc.update(date(2026, 11, 24), games, before)
    assert [p["name"] for p in d["players"]] == ["Big", "Mid"]    # under 18 projected is left off
    # after tip, a changed projection does not move the frozen numbers
    games[0]["players"]["home"][1]["points"] = 60
    after = datetime(2026, 11, 25, 1, 0, tzinfo=timezone.utc)
    d2 = kotc.update(date(2026, 11, 24), games, after)
    assert d2["players"][0]["name"] == "Big" and d2["players"][0]["chance"] == d["players"][0]["chance"]
    # grading: "Other" (not on our list) leads the night
    box = pd.DataFrame({"game_id": ["1", "1", "1"], "athlete_id": [10, 11, 99],
                        "athlete_display_name": ["Big", "Mid", "Other"], "date": pd.to_datetime(["2026-11-24"] * 3),
                        "points": [30, 20, 40], "rebounds": [10, 5, 10], "assists": [5, 5, 8]})
    r = kotc.grade(box)
    assert r["nights"] == 1 and r["top_pick_won"] == 0 and r["leader_in_top10"] == 0
    assert r["recent"][0]["leader"] == "Other" and r["recent"][0]["pick_pra"] == 45
    box.loc[2, "points"] = 1
    r = kotc.grade(box)
    assert r["top_pick_won"] == 1 and r["recent"][0]["leader_rank"] == 1
