"""College player projections: the NBA method (nbamodel/players.py) on Division I box scores.

Games are 40 minutes (200 team minutes). Only games between two Division I teams are used, the same
as the team ratings. hoopR's college box scores are missing minutes for some games; a team's game
whose minutes don't add up to a full game is dropped.

    python -m cbbmodel.players      # backtest: writes docs/cbb/players_backtest.json
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nbamodel import players as P

from . import data

DOCS = Path(__file__).resolve().parent.parent / "docs" / "cbb"


def box(season: int, refresh: bool = False) -> pd.DataFrame:
    p = pd.read_parquet(data._get(f"player_box/parquet/player_box_{season}.parquet", data.CACHE / f"player_box_{season}.parquet", refresh))
    p = p[p.season_type.isin([2, 3])].copy()
    d1 = data.division_one(data.schedule(season, refresh=False))
    p = p[pd.to_numeric(p.team_id, errors="coerce").isin(d1) & pd.to_numeric(p.opponent_team_id, errors="coerce").isin(d1)]
    b = P.clean_box(p)
    # keep a game only when both teams' minutes look complete (190+ of 200, allowing for rounding)
    tm = b.groupby(["game_id", "team_id"]).minutes.sum()
    ok = tm[tm >= 190].reset_index().groupby("game_id").team_id.nunique()
    return b[b.game_id.isin(ok[ok == 2].index)].reset_index(drop=True)


CBB = P.League("cbb", box, team_min=200.0, max_min=39.0, tune=[2017, 2018, 2019], grade=[2023, 2024, 2025, 2026], docs=DOCS)


if __name__ == "__main__":
    P.run(CBB)
