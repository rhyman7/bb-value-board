"""The line log: every ESPN line the pipelines read, with the model's number at that moment.

Each pipeline run reads ESPN's scoreboard for the next few days and appends one row per game that
hasn't tipped off: the spread (+ = home favored), the total, and the model's margin and total as
the pipeline projected them in the same run. Over a season the log holds each game's line history:

- the first row for a game is the earliest line the board saw (the opener, or close to it);
- the last row before tip-off stands in for the close.

`openers.py` grades the model against both. The NBA pipeline also uses the last rows as this
season's closes for its market ratings, since hoopR's closing lines arrive after the season.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = {"nba": ROOT / "docs" / "lines" / "nba_lines.csv", "cbb": ROOT / "docs" / "lines" / "cbb_lines.csv"}
SCOREBOARD = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d}",
    "cbb": "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d}&groups=50&limit=400",
}
COLUMNS = ["game_id", "fetched_at", "close_spread", "close_total", "model_margin", "model_total", "home_ml", "away_ml", "model_win_prob"]


def _json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "bb-value-board"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def home_spread(details, home, away):
    """ESPN's 'BOS -6.5' as the home team's expected margin (+ = home favored)."""
    det = (details or "").strip()
    if det.lower().startswith(("even", "pk", "pick")):
        return 0.0
    if " " not in det:
        return None
    ab, pts = det.rsplit(" ", 1)
    try:
        pts = abs(float(pts))
    except ValueError:
        return None
    return pts if ab == home else -pts if ab == away else None


def _ml(side_odds, odds, side):
    """American moneyline for one side: ESPN has used both homeTeamOdds.moneyLine and
    moneyline.home.close.odds ("-150" as text)."""
    v = (side_odds or {}).get("moneyLine")
    if v is None:
        m = ((odds.get("moneyline") or {}).get(side) or {})
        v = ((m.get("close") or {}).get("odds")) or ((m.get("open") or {}).get("odds"))
    try:
        v = float(str(v).replace("+", "")) if v not in (None, "", "OFF") else None
    except ValueError:
        v = None
    return v if v is None or abs(v) >= 100 else None


def espn_lines(league, days):
    """{game_id: (spread, total, home_ml, away_ml)} for games that haven't started, from ESPN's
    scoreboard. Empty when ESPN can't be reached; a missing piece is None."""
    out = {}
    for d in days:
        try:
            js = _json(SCOREBOARD[league].format(d=d.strftime("%Y%m%d")))
        except Exception:
            continue
        for ev in js.get("events", []):
            comp = (ev.get("competitions") or [{}])[0]
            state = (((comp.get("status") or ev.get("status") or {}).get("type")) or {}).get("state")
            odds = (comp.get("odds") or [None])[0]
            if not odds or state not in (None, "pre"):
                continue
            teams = {c.get("homeAway"): (c.get("team") or {}).get("abbreviation", "") for c in comp.get("competitors", [])}
            sp = home_spread(odds.get("details"), teams.get("home"), teams.get("away"))
            tot = odds.get("overUnder")
            out[str(ev.get("id"))] = (sp, float(tot) if tot is not None else None,
                                      _ml(odds.get("homeTeamOdds"), odds, "home"), _ml(odds.get("awayTeamOdds"), odds, "away"))
    return out


def log(league, games):
    """Append rows for games with a line: dicts with id, mkt_spread, mkt_total, margin, total."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [{"game_id": str(x["id"]), "fetched_at": now, "close_spread": x.get("mkt_spread"),
             "close_total": x.get("mkt_total"), "model_margin": x.get("margin"), "model_total": x.get("total"),
             "home_ml": x.get("mkt_ml_home"), "away_ml": x.get("mkt_ml_away"), "model_win_prob": x.get("win_prob")}
            for x in games if x.get("mkt_spread") is not None or x.get("mkt_ml_home") is not None]
    if not rows:
        return 0
    path = LOG[league]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, nrows=0)
        if list(old.columns) != COLUMNS:          # an older log without the moneyline columns
            pd.read_csv(path).reindex(columns=COLUMNS).to_csv(path, index=False)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
    return len(rows)


def read(league):
    path = LOG[league]
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={"game_id": str}).sort_values("fetched_at")


def last_lines(league):
    """The last logged line per game (stands in for the close)."""
    return read(league).drop_duplicates("game_id", keep="last")[["game_id", "close_spread", "close_total"]]
