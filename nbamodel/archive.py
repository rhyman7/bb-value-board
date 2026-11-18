"""Past nights: every game on the board frozen as it stood before tip-off, then graded once the
final is in. Shared by the NBA and college pipelines.

    freeze(league, games, now)   save today's games that have not tipped yet (a later run the same
                                 day replaces them, so each game keeps its last pre-tip numbers)
    grade(league, results)       fill in finals and write the season record (index.json)

Files: docs/archive/<league>/<date>.json and docs/archive/<league>/index.json.
Spreads and margins are home minus away (+ = home favored), as in the feeds.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "docs" / "archive"
KEEP = ["id", "date", "tip", "away", "home", "neutral", "margin", "total", "win_prob", "projection",
        "mkt_spread", "mkt_total", "mkt_ml_home", "mkt_ml_away"]


def _path(league, day):
    return ROOT / league / f"{day}.json"


def _read(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def freeze(league, games, now: datetime):
    """games: the feed's game list (each with 'tip', an ISO time with offset). Only games dated
    today in the feed are frozen, and only before they tip."""
    today = now.date().isoformat()
    todays = [g for g in games if g.get("date") == today and g.get("tip")]
    if not todays:
        return 0
    p = _path(league, today)
    night = _read(p) or {"date": today, "games": {}}
    n = 0
    for g in todays:
        if datetime.fromisoformat(g["tip"]) <= now:
            continue                      # already started: keep what was frozen before tip-off
        night["games"][g["id"]] = {**{k: g.get(k) for k in KEEP}, "frozenAt": now.isoformat(timespec="minutes")}
        n += 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(night, indent=1, allow_nan=False))
    return n


def _res(a, b):
    """W/L/P for taking side a against line b's direction: sign(a) is the outcome vs the line,
    sign(b) the model's lean."""
    if b == 0:
        return None
    s = np.sign(a) * np.sign(b)
    return "W" if s > 0 else "L" if s < 0 else "P"


def grade_game(g, pts_h, pts_a):
    margin, total = pts_h - pts_a, pts_h + pts_a
    g["final"] = {"home": int(pts_h), "away": int(pts_a)}
    g["su"] = None if margin == 0 or g.get("margin") is None else ("W" if np.sign(margin) == np.sign(g["margin"]) else "L")
    sp, tot = g.get("mkt_spread"), g.get("mkt_total")
    g["ats"] = _res(margin - sp, g["margin"] - sp) if sp is not None and g.get("margin") is not None else None
    g["ou"] = _res(total - tot, g["total"] - tot) if tot is not None and g.get("total") is not None else None
    g["err"] = {"model": round(abs(margin - g["margin"]), 1) if g.get("margin") is not None else None,
                "line": round(abs(margin - sp), 1) if sp is not None else None}
    return g


def _rec(gs, key):
    r = [g[key] for g in gs if g.get(key)]
    w, l, p = r.count("W"), r.count("L"), r.count("P")
    return {"w": w, "l": l, "p": p, "pct": round(100 * w / (w + l), 1) if w + l else None}


def summarize(gs):
    graded = [g for g in gs if g.get("final")]
    both = [g for g in graded if g["err"]["line"] is not None and g["err"]["model"] is not None]
    return {"games": len(gs), "graded": len(graded), "su": _rec(graded, "su"), "ats": _rec(graded, "ats"), "ou": _rec(graded, "ou"),
            "model_mae": round(float(np.mean([g["err"]["model"] for g in both])), 2) if both else None,
            "line_mae": round(float(np.mean([g["err"]["line"] for g in both])), 2) if both else None,
            "with_line": len(both)}


def grade(league, results, season_start: str):
    """results: DataFrame of finished games with game_id, pts_h, pts_a. Writes index.json with each
    night's record and the season's (nights on or after season_start)."""
    d = ROOT / league
    if not d.exists():
        return None
    fin = {str(r.game_id): (int(r.pts_h), int(r.pts_a)) for r in results.itertuples()}
    nights, season = [], []
    for p in sorted(d.glob("????-??-??.json")):
        night = _read(p)
        if not night:
            continue
        changed = False
        for gid, g in night["games"].items():
            if not g.get("final") and gid in fin:
                grade_game(g, *fin[gid])
                changed = True
        if changed:
            p.write_text(json.dumps(night, indent=1, allow_nan=False))
        gs = list(night["games"].values())
        nights.append({"date": night["date"], **summarize(gs)})
        if night["date"] >= season_start:
            season.extend(gs)
    idx = {"generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"), "seasonStart": season_start,
           "season": summarize(season), "nights": nights[::-1]}
    (d / "index.json").write_text(json.dumps(idx, indent=1, allow_nan=False))
    return idx
