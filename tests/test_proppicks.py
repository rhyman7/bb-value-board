from datetime import date

import numpy as np

from nbamodel import propprob, proppicks
from oddsapi.props import _best


def test_chances_sum_to_one_and_push_on_whole_lines():
    po, pu, pp = propprob.chances(np.array([2.0, 2.0, 25.0]), np.array([1.5, 2.0, 24.5]), 0.05)
    assert np.allclose(po + pu + pp, 1)
    assert pp[0] == 0 and pp[1] > 0 and pp[2] == 0


def test_best_price_skips_offshore_books():
    b = _best([(24.5, -110, -110, "draftkings"), (24.5, -105, -115, "fanduel"), (24.5, 120, 120, "bovada")])
    assert b == {"best_over": -105, "best_over_book": "fanduel", "best_under": -110, "best_under_book": "draftkings"}


def test_picks_need_edge_at_best_price():
    row = {"id": 1, "name": "A", "team": "NY", "opp": "BOS", "points": 20.5, "rebounds": 5.0, "assists": 3.0, "threes": 2.6,
           "props": {"threes": {"line": 2.5, "over": -140, "under": 110, "books": 7, "best_over": -130,
                                "best_over_book": "betmgm", "best_under": 120, "best_under_book": "williamhill_us"},
                     "points": {"line": 20.5, "over": -110, "under": -110, "books": 7, "best_over": -110,
                                "best_over_book": "fanduel", "best_under": -110, "best_under_book": "fanduel"}}}
    out = proppicks.picks(date(2026, 11, 19), {"1": {"home": [row], "away": []}})
    assert [(p["stat"], p["side"], p["book"]) for p in out] == [("threes", "under", "Caesars")]
    assert all(p["edge"] >= 0.03 for p in out)
