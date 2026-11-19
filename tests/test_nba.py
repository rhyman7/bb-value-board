import numpy as np
import pandas as pd

from nbamodel import data, model, ratings


def _games():
    # three teams, A clearly best, C worst, alternating home court
    rows = []
    day = pd.Timestamp("2026-11-01")
    pairs = [("A", "B", 115, 100), ("B", "C", 112, 104), ("C", "A", 98, 118)] * 6
    for i, (h, a, ph, pa) in enumerate(pairs):
        rows.append({"game_id": str(i), "season": 2027, "date": day + pd.Timedelta(days=i), "home_id": h, "away_id": a,
                     "home": h, "away": a, "pts_h": ph, "pts_a": pa, "poss": 100.0, "neutral": False})
    g = pd.DataFrame(rows)
    g["margin"] = g.pts_h - g.pts_a
    g["total"] = g.pts_h + g.pts_a
    return g


def test_ratings_order_teams():
    g = _games()
    R, mean, hca, pace = ratings.rate_day(g, pd.Timestamp("2026-12-01"), 2027, ratings.Settings())
    net = {t: o + d for t, (o, d, p) in R.items()}
    assert net["A"] > net["B"] > net["C"]


def test_rest_counts_back_to_backs():
    g = _games()
    g = data.add_rest(g)
    # every team plays every 1-2 days in the toy schedule; the first game of a season counts as rested
    assert g.rest_h.min() >= 0 and g.rest_h.max() <= 4
    assert (g.iloc[0][["rest_h", "rest_a"]] == 4).all()


def test_market_weight_recovers_signal():
    rng = np.random.default_rng(0)
    line = rng.normal(0, 6, 5000)
    model_ = line + rng.normal(0, 3, 5000)
    actual = line + 0.4 * (model_ - line) + rng.normal(0, 11, 5000)
    df = pd.DataFrame({"m": model_, "l": line, "a": actual})
    w, se, n = model.market_weight(df, "m", "l", "a")
    assert abs(w - 0.4) < 3 * se


def test_espn_athlete_id_from_injury_links():
    from nbamodel.availability import espn_athlete_id
    # ESPN's injuries feed (Nov 2026) has no athlete.id, only links
    ath = {"displayName": "Jock Landale", "links": [
        {"rel": ["playercard"], "href": "https://www.espn.com/nba/player/_/id/3146557/jock-landale"}]}
    assert espn_athlete_id(ath) == 3146557
    assert espn_athlete_id({"links": [{"href": "sportscenter://x?uid=s:40~l:46~a:42&section=stats"}]}) == 42
    assert espn_athlete_id({"id": "7"}) == 7
    assert espn_athlete_id({}) is None and espn_athlete_id(None) is None


def test_injuries_parse_feed_without_athlete_id(monkeypatch):
    from nbamodel import pipeline
    feed = {"injuries": [{"id": "1", "injuries": [
        {"status": "Out", "athlete": {"links": [{"href": "https://www.espn.com/nba/player/_/id/111/a"}]}},
        {"status": "Day-To-Day", "athlete": {"links": [{"href": "https://www.espn.com/nba/player/_/id/222/b"}]}},
    ]}]}
    monkeypatch.setattr(pipeline, "_json", lambda url: feed)
    assert pipeline.injuries() == {111: "out"}


def test_median_price_stays_a_price():
    from oddsapi.props import median_price
    assert median_price([-110, 105]) in (-103, -102)   # plain median would be -2.5
    assert median_price([-110, -110, -120]) == -110
    assert median_price([120, 140]) == 130
    assert median_price([]) is None
