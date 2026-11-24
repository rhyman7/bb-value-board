"""DraftKings "King of the Court": each player's chance to lead the whole night's slate in
points + rebounds + assists (PRA).

Each player's PRA is drawn from a normal around tonight's projection (players.tonight) with a spread
that grows with the projection, plus a shared shock per game (pace, overtime). Ties split the prize,
so a tie counts as a share. The spread, SD = 0.15 x projection + 6.5, was set on 475 past nights
(2024-26, walk-forward projections, players who sat left out): the top pick was given 26% on average
and won 28%. A plain "highest projection" pick won 29% of those nights.

Writes docs/nba/kotc.json for the board and logs the top ten each night to docs/lines/nba_kotc.csv.
Once the first game of the night tips, the night's numbers are frozen. The morning run grades past
nights against the box scores (all players who played, not just the ones on our list).
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "nba" / "kotc.json"
LOG = ROOT / "docs" / "lines" / "nba_kotc.csv"

SD_SLOPE, SD_BASE = 0.15, 6.5   # PRA spread around the projection (fit 2024-26, see module doc)
GAME_SHOCK = 0.06               # shared scoring swing per game, as a share of each projection
MIN_PROJ = 18.0                 # players projected below this never lead; skipped for speed
SIMS = 100_000
SHOW = 10
BACKTEST = ("Set on 475 past nights (2024-26): the top pick was given 26% on average and won 28%. "
            "A plain highest-projection pick won 29%, so this is a probability guide, not an edge.")
LOG_COLS = ["date", "rank", "game_id", "athlete_id", "name", "team", "opp", "tip", "proj", "chance"]


def chances(mu, game, sims=SIMS, seed=0):
    """mu: projected PRA per player; game: game id per player. Returns (chance each player leads,
    with ties split; median winning PRA; share of nights with a tie at the top)."""
    mu = np.asarray(mu, float)
    gi, _ = pd.factorize(pd.Series(game, dtype=str))
    rng = np.random.default_rng(seed)
    out = np.zeros(len(mu))
    wins, ties = [], 0
    for n in [50_000] * (sims // 50_000) + ([sims % 50_000] if sims % 50_000 else []):
        shock = rng.normal(0, GAME_SHOCK, (n, gi.max() + 1))[:, gi]
        d = np.round(np.maximum(0, mu * (1 + shock) + rng.normal(0, 1, (n, len(mu))) * (SD_SLOPE * mu + SD_BASE)))
        top = d.max(1, keepdims=True)
        w = d == top
        k = w.sum(1)
        out += (w / k[:, None]).sum(0)
        wins.append(top[:, 0])
        ties += int((k > 1).sum())
    return out / sims, float(np.median(np.concatenate(wins))), ties / sims


def tonight(day, games):
    """games: the feed's games (with "players"). Today's games only. Returns the board's dict or None."""
    todays = [g for g in games if g["date"] == day.isoformat() and g.get("players")]
    rows = []
    for g in todays:
        for side in ("home", "away"):
            for p in g["players"].get(side) or []:
                mu = p["points"] + p["rebounds"] + p["assists"]
                if mu < MIN_PROJ:
                    continue
                avg = p.get("avg") or {}
                a = [avg.get(k) for k in ("points", "rebounds", "assists")]
                rows.append({"game_id": g["id"], "athlete_id": p["id"], "name": p["name"], "team": p["team"], "opp": p["opp"],
                             "tip": g["tip"], "proj": round(mu, 1), "pts": p["points"], "reb": p["rebounds"], "ast": p["assists"],
                             "avg": round(sum(a), 1) if None not in a else None,
                             "out_team": (g["players"].get("out") or {}).get(side, [])})
    if not rows:
        return None
    ch, med, tie = chances([r["proj"] for r in rows], [r["game_id"] for r in rows])
    for r, c in zip(rows, ch):
        r["chance"] = round(float(c), 4)
    rows.sort(key=lambda r: -r["chance"])
    return {"date": day.isoformat(), "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first_tip": min(g["tip"] for g in todays),
            "games": [{"id": g["id"], "away": g["away"], "home": g["home"], "tip": g["tip"],
                       "out": {"away": g["players"]["out"].get("away", []), "home": g["players"]["out"].get("home", [])}}
                      for g in sorted(todays, key=lambda g: g["tip"])],
            "median_winning_pra": med, "tie_rate": round(tie, 3), "players": rows[:SHOW]}


def log(night):
    new = pd.DataFrame([{**{k: r[k] for k in LOG_COLS if k in r}, "date": night["date"], "rank": i + 1}
                        for i, r in enumerate(night["players"])], columns=LOG_COLS)
    if LOG.exists():
        old = pd.read_csv(LOG, dtype={"game_id": str})
        new = pd.concat([old[old.date != night["date"]], new], ignore_index=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["date", "rank"]).to_csv(LOG, index=False)


def grade(box):
    """box: players.box(season), one row per player who played. Grades every logged night whose
    games are all in the box scores. Returns the record, or None before the first graded night."""
    if not LOG.exists():
        return None
    lg = pd.read_csv(LOG, dtype={"game_id": str})
    b = box.copy()
    b["pra"] = b.points + b.rebounds + b.assists
    b["day"] = b.date.dt.strftime("%Y-%m-%d")
    have = set(b.game_id)
    nights = []
    for day, n in lg.groupby("date"):
        if not set(n.game_id) <= have:
            continue    # not every game is in yet
        bd = b[b.day == day]
        if bd.empty:
            continue
        best = bd.pra.max()
        lead = bd[bd.pra == best]
        share = 1.0 / len(lead)
        top = n.sort_values("rank").iloc[0]
        got = bd[bd.athlete_id == top.athlete_id]
        nights.append({"date": day, "pick": top["name"], "pick_chance": float(top.chance),
                       "pick_pra": int(got.pra.iloc[0]) if len(got) else None,
                       "leader": " / ".join(lead.athlete_display_name), "leader_pra": int(best),
                       "won": share if int(top.athlete_id) in set(lead.athlete_id) else 0.0,
                       "leader_rank": int(n[n.athlete_id.isin(lead.athlete_id)]["rank"].min()) if n.athlete_id.isin(lead.athlete_id).any() else None})
    if not nights:
        return None
    return {"nights": len(nights), "top_pick_won": round(sum(x["won"] for x in nights), 2),
            "expected": round(sum(x["pick_chance"] for x in nights), 2),
            "leader_in_top10": sum(1 for x in nights if x["leader_rank"] is not None),
            "recent": sorted(nights, key=lambda x: x["date"], reverse=True)[:10]}


def update(day, games, now, box=None):
    """Called from the board pipeline. Tonight's numbers refresh on every run until the first tip,
    then stay as they were. Past nights are graded from box (if given)."""
    prev = json.loads(OUT.read_text()) if OUT.exists() else None
    night = None
    if prev and prev.get("date") == day.isoformat() and datetime.fromisoformat(prev["first_tip"]) <= now:
        night = {k: v for k, v in prev.items() if k not in ("record", "backtest")}   # frozen at first tip
    else:
        night = tonight(day, games)
        if night:
            log(night)
    record = None
    if box is not None:
        try:
            record = grade(box)
        except Exception as e:
            print("king of the court grade failed:", repr(e))
    doc = night or {"date": day.isoformat(), "players": [], "games": []}
    doc["record"] = record if record is not None else (prev or {}).get("record")
    doc["backtest"] = BACKTEST
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, allow_nan=False))
    return doc
