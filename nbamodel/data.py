"""Game data for the NBA model: team box scores from sportsdataverse's hoopR NBA data, one row
per game with both teams' possessions and points, plus consensus closing lines (The Odds API
backfill hoopR publishes, from the 2020 bubble on).

Season numbers follow hoopR: the year the season ends (2026-27 is 2027).
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
FIRST_SEASON = 2015


def _get(rel: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": "bb-value-board"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                dest.write_bytes(r.read())
            return dest
        except Exception:
            if attempt == 3:
                if dest.exists():
                    return dest
                raise
            time.sleep(10 * (attempt + 1))
    return dest


def team_box(season: int, refresh: bool = False) -> pd.DataFrame:
    return pd.read_parquet(_get(f"team_box/parquet/team_box_{season}.parquet", CACHE / f"team_box_{season}.parquet", refresh))


def schedule(season: int, refresh: bool = False) -> pd.DataFrame:
    return pd.read_parquet(_get(f"schedules/parquet/nba_schedule_{season}.parquet", CACHE / f"nba_schedule_{season}.parquet", refresh))


def closing_lines(refresh: bool = False) -> pd.DataFrame:
    c = pd.read_parquet(_get("betting_lines/closing_lines_odds_api.parquet", CACHE / "closing_lines.parquet", refresh))
    c = c.rename(columns={"over_under": "close_total"})
    # home_point: the home team's spread, Vegas sign (negative = home favored). Our sign: + = home favored.
    c["close_spread"] = -c["home_point"]
    c["game_id"] = c["game_id"].astype(str)
    return c[["game_id", "close_spread", "close_total", "n_books"]]


def possessions(fga, oreb, tov, fta):
    return fga - oreb + tov + 0.44 * fta


def games(season: int, refresh: bool = False) -> pd.DataFrame:
    """One row per regular-season game: home and away team ids/abbreviations, points, possessions."""
    tb = team_box(season, refresh)
    tb = tb[tb.season_type == 2].copy()
    # drop exhibition teams (All-Star and the like): keep teams with a real schedule
    counts = tb.team_id.value_counts()
    real = set(counts[counts >= 0.5 * counts.median()].index)
    tb = tb[tb.team_id.isin(real) & tb.opponent_team_id.isin(real)]
    for c in ["team_score", "field_goals_attempted", "offensive_rebounds", "free_throws_attempted", "total_turnovers", "turnovers"]:
        tb[c] = pd.to_numeric(tb[c], errors="coerce").fillna(0)
    tov = tb.total_turnovers.where(tb.total_turnovers > 0, tb.turnovers)
    tb["poss"] = possessions(tb.field_goals_attempted, tb.offensive_rebounds, tov, tb.free_throws_attempted)
    h = tb[tb.team_home_away == "home"]
    a = tb[tb.team_home_away == "away"]
    keep = ["game_id", "team_id", "team_abbreviation", "team_score", "poss"]
    g = h[keep + ["game_date"]].merge(a[keep], on="game_id", suffixes=("_h", "_a"))
    g = g.rename(columns={
        "team_id_h": "home_id", "team_id_a": "away_id", "team_abbreviation_h": "home", "team_abbreviation_a": "away",
        "team_score_h": "pts_h", "team_score_a": "pts_a",
    })
    g["game_id"] = g["game_id"].astype(str)
    g["date"] = pd.to_datetime(g["game_date"])
    g["season"] = season
    g["poss"] = (g.poss_h + g.poss_a) / 2
    g["margin"] = g.pts_h - g.pts_a
    g["total"] = g.pts_h + g.pts_a
    # the 2020 restart in the Orlando bubble: no home court
    g["neutral"] = (season == 2020) & (g.date >= "2020-07-01")
    return g.drop(columns=["game_date", "poss_h", "poss_a"]).sort_values(["date", "game_id"]).reset_index(drop=True)


def all_games(last_season: int, first_season: int = FIRST_SEASON, refresh_current: bool = True) -> pd.DataFrame:
    parts = [games(s, refresh=(refresh_current and s == last_season)) for s in range(first_season, last_season + 1)]
    g = pd.concat(parts, ignore_index=True)
    g = add_rest(g)
    return g


def add_rest(g: pd.DataFrame) -> pd.DataFrame:
    """Days off before each game for both teams (0 = second night of a back-to-back, capped at 4;
    a team's first game of a season counts as 4) and games played in the 6 days before."""
    long = pd.concat([
        g[["game_id", "season", "date", "home_id"]].rename(columns={"home_id": "tid"}).assign(side="h"),
        g[["game_id", "season", "date", "away_id"]].rename(columns={"away_id": "tid"}).assign(side="a"),
    ], ignore_index=True).sort_values(["season", "tid", "date"]).reset_index(drop=True)
    long["prev"] = long.groupby(["season", "tid"])["date"].shift(1)
    long["rest"] = ((long.date - long.prev).dt.days - 1).clip(lower=0, upper=4).fillna(4)
    g7 = np.zeros(len(long), dtype=int)
    for _, ix in long.groupby(["season", "tid"]).indices.items():
        d = long.date.values[ix]
        lo = np.searchsorted(d, d - np.timedelta64(6, "D"), side="left")
        g7[ix] = np.arange(len(d)) - lo
    long["g7"] = g7
    piv = long.pivot_table(index="game_id", columns="side", values=["rest", "g7"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return g.merge(piv.reset_index(), on="game_id", how="left")
