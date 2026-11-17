"""Game data for the college model: men's Division I games from sportsdataverse's hoopR college
data (team box scores and schedules), one row per game with both teams' points and possessions,
plus ESPN's spread for the game, which hoopR keeps in its play-by-play files (2015 on; the
2025-26 season's files carry none).

Division I teams are the ones ESPN gives a conference that season. Only games between two
Division I teams are kept. Season numbers are the year the season ends (2026-27 is 2027).
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "mbb"
FIRST_SEASON = 2015


def _get(rel: str, dest: Path, refresh: bool) -> Path:
    if dest.exists() and not refresh:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(f"{BASE}/{rel}", headers={"User-Agent": "bb-value-board"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                dest.write_bytes(r.read())
            return dest
        except Exception:
            if attempt == 3:
                if dest.exists():
                    return dest
                raise
            time.sleep(10 * (attempt + 1))
    return dest


def team_box(season, refresh=False):
    return pd.read_parquet(_get(f"team_box/parquet/team_box_{season}.parquet", CACHE / f"team_box_{season}.parquet", refresh))


def schedule(season, refresh=False):
    return pd.read_parquet(_get(f"schedules/parquet/mbb_schedule_{season}.parquet", CACHE / f"mbb_schedule_{season}.parquet", refresh))


def spreads(season, refresh=False):
    """ESPN's spread per game (+ = home favored), from the play-by-play file. Cached as a small
    extract; the play-by-play file itself (about 60 MB) is only read once."""
    small = CACHE / f"mbb_spreads_{season}.parquet"
    if not small.exists() or refresh:
        big = _get(f"pbp/parquet/play_by_play_{season}.parquet", CACHE / f"pbp_{season}.parquet", True)
        d = pd.read_parquet(big, columns=["game_id", "game_spread", "home_favorite", "game_spread_available", "home_team_spread"])
        d.drop_duplicates("game_id").to_parquet(small)
        big.unlink(missing_ok=True)
    d = pd.read_parquet(small)
    d = d[d.game_spread_available.fillna(False).astype(bool)]
    # home_team_spread is the home team's expected margin in every season's files
    return pd.DataFrame({"game_id": d.game_id.astype(str), "close_spread": d.home_team_spread.astype(float)})


def division_one(sched):
    ids = set()
    for side in ("home", "away"):
        ids |= set(sched.loc[sched[f"{side}_conference_id"].notna(), f"{side}_id"].astype(int))
    return ids


def possessions(fga, oreb, tov, fta):
    return fga - oreb + tov + 0.475 * fta     # college free throws: 0.475 per attempt (KenPom)


def games(season, refresh=False):
    sched = schedule(season, refresh)
    d1 = division_one(sched)
    tb = team_box(season, refresh)
    tb = tb[tb.season_type.isin([2, 3])].copy()
    tb["team_id"] = tb.team_id.astype(int)
    tb["opponent_team_id"] = tb.opponent_team_id.astype(int)
    tb = tb[tb.team_id.isin(d1) & tb.opponent_team_id.isin(d1)]
    for c in ["team_score", "field_goals_attempted", "offensive_rebounds", "free_throws_attempted", "total_turnovers", "turnovers"]:
        tb[c] = pd.to_numeric(tb[c], errors="coerce").fillna(0)
    tov = tb.total_turnovers.where(tb.total_turnovers > 0, tb.turnovers)
    tb["poss"] = possessions(tb.field_goals_attempted, tb.offensive_rebounds, tov, tb.free_throws_attempted)
    h = tb[tb.team_home_away == "home"]
    a = tb[tb.team_home_away == "away"]
    keep = ["game_id", "team_id", "team_abbreviation", "team_score", "poss"]
    g = h[keep + ["game_date", "season_type"]].merge(a[keep], on="game_id", suffixes=("_h", "_a"))
    g = g.rename(columns={"team_id_h": "home_id", "team_id_a": "away_id", "team_abbreviation_h": "home",
                          "team_abbreviation_a": "away", "team_score_h": "pts_h", "team_score_a": "pts_a"})
    g["game_id"] = g.game_id.astype(str)
    g["date"] = pd.to_datetime(g.game_date)
    g["season"] = season
    g["poss"] = (g.poss_h + g.poss_a) / 2
    g = g[(g.poss > 40) & (g.pts_h > 0) & (g.pts_a > 0)]       # box scores with missing stats
    g["margin"] = g.pts_h - g.pts_a
    g["total"] = g.pts_h + g.pts_a
    s = sched[["game_id", "neutral_site", "conference_competition", "venue_address_state"]].copy()
    s["game_id"] = s.game_id.astype(str)
    g = g.merge(s, on="game_id", how="left")
    g["neutral"] = g.neutral_site.fillna(False).astype(bool)
    g["conf_game"] = g.conference_competition.fillna(False).astype(bool)
    g = g.drop(columns=["game_date", "poss_h", "poss_a", "neutral_site", "conference_competition"])
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)


def all_games(last_season, first_season=FIRST_SEASON, refresh_current=True, refresh_spreads=False):
    parts = []
    for s in range(first_season, last_season + 1):
        x = games(s, refresh=(refresh_current and s == last_season))
        try:
            x = x.merge(spreads(s, refresh=(refresh_spreads and s == last_season)), on="game_id", how="left")
        except Exception:
            x["close_spread"] = np.nan
        parts.append(x)
    g = pd.concat(parts, ignore_index=True)
    from nbamodel.data import add_rest
    return add_rest(g)
