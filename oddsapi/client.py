"""The Odds API (the-odds-api.com), v4. The key is the ODDS_API_KEY repository secret, so this only
runs from GitHub Actions. Every call prints the credits it used and how many remain.

Costs (from the-odds-api.com/historical-odds-data):
  /sports                         free
  live odds                       markets x regions per call
  live event odds (props)         markets x regions per event
  historical odds (featured)      10 x markets x regions per snapshot
  historical event odds (props)   10 x markets x regions per event
  historical events               1 per call
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

BASE = "https://api.the-odds-api.com/v4"
SPORTS = {"nba": "basketball_nba", "cbb": "basketball_ncaab"}
used_total = 0


def get(path, **params):
    """GET with the key; returns (json, headers). Never prints the key."""
    global used_total
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set")
    q = urllib.parse.urlencode({**params, "apiKey": key})
    req = urllib.request.Request(f"{BASE}{path}?{q}", headers={"User-Agent": "bb-value-board"})
    with urllib.request.urlopen(req, timeout=30) as r:
        h = {k.lower(): v for k, v in r.headers.items()}
        body = json.load(r)
    last = int(h.get("x-requests-last", 0) or 0)
    used_total += last
    print(f"  {path} cost {last}, used {h.get('x-requests-used')}, remaining {h.get('x-requests-remaining')}")
    return body, h


def remaining():
    _, h = get("/sports")
    return int(h.get("x-requests-remaining", 0)), int(h.get("x-requests-used", 0))
