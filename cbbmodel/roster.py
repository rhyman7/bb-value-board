"""Roster turnover: how much of a team's production comes back, and what transfers bring.

College rosters change a lot between seasons, so last season's rating is only a starting point.
For each player and season, value = Game Score above a replacement level over his minutes, per
team game (about a point of margin per unit). A team's roster strength for a season is the sum of
last season's values of the players on this season's roster, wherever they played last season
(freshmen and players new to Division I count as replacement level). The prior shift is roster
strength minus what last season's team produced: positive when the team returns or adds more than
it had, negative when it loses its best players.

This season's roster is the players who appear in the team's box scores that season, which a
preseason roster gives almost exactly (the exception is a player hurt for the whole season).
"""
from __future__ import annotations

import pandas as pd

from .data import CACHE, _get

REPLACEMENT = 0.25     # Game Score per minute of a replacement-level college player


def player_box(season, refresh=False):
    p = pd.read_parquet(_get(f"player_box/parquet/player_box_{season}.parquet", CACHE / f"player_box_{season}.parquet", refresh),
                        columns=None)
    p = p[p.season_type.isin([2, 3])].copy()
    num = ["minutes", "points", "field_goals_made", "field_goals_attempted", "free_throws_made", "free_throws_attempted",
           "offensive_rebounds", "defensive_rebounds", "steals", "assists", "blocks", "fouls", "turnovers"]
    for c in num:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    p["gmsc"] = (p.points + 0.4 * p.field_goals_made - 0.7 * p.field_goals_attempted
                 - 0.4 * (p.free_throws_attempted - p.free_throws_made) + 0.7 * p.offensive_rebounds
                 + 0.3 * p.defensive_rebounds + p.steals + 0.7 * p.assists + 0.7 * p.blocks
                 - 0.4 * p.fouls - p.turnovers)
    p["team_id"] = pd.to_numeric(p.team_id, errors="coerce")
    p["athlete_id"] = pd.to_numeric(p.athlete_id, errors="coerce")
    return p.dropna(subset=["team_id", "athlete_id"])[["game_id", "team_id", "athlete_id", "minutes", "gmsc"]]


def season_values(season):
    """{athlete: value per team game}, and {team: total value per game} for one season."""
    p = player_box(season)
    p = p[p.minutes > 0]
    team_games = p.groupby("team_id").game_id.nunique()
    by = p.groupby(["athlete_id", "team_id"])[["gmsc", "minutes"]].sum().reset_index()
    by["value"] = (by.gmsc - REPLACEMENT * by.minutes) / by.team_id.map(team_games)
    # a player on two teams in one season: his values add
    player = by.groupby("athlete_id").value.sum()
    team = by.groupby("team_id").value.sum()
    return player, team


def prior_shift(seasons):
    """{season: {team_id: roster strength minus last season's team value}} (per game, Game Score units)."""
    out = {}
    cache = {}
    for s in seasons:
        try:
            if s - 1 not in cache:
                cache[s - 1] = season_values(s - 1)
            pv, tv = cache[s - 1]
            p = player_box(s)
        except Exception:
            continue
        roster = p[p.minutes > 0].drop_duplicates(["team_id", "athlete_id"])
        strength = roster.assign(v=roster.athlete_id.map(pv).fillna(0.0)).groupby("team_id").v.sum()
        shift = (strength - tv.reindex(strength.index).fillna(0.0))
        # each team's shift measured against the league's (graduations make the average negative)
        shift = shift - shift.mean()
        out[s] = shift.to_dict()
    return out
