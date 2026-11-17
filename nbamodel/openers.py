"""The opening-line test: does the model beat the first line, before the market has moved?

From the line log (lines.py), for every finished game:

- the opener is the first line logged for the game, with the model's number from the same run;
- the close is the last line logged before tip-off.

Three questions, each for side and total, by how far the model was off the opener:

1. Record: betting the model's side at the opener (−110).
2. Line movement (closing line value): did the line move toward the model between the opener and
   the close? A model that keeps getting the line to move its way is ahead of the market, even
   before enough games have finished to show it in the record.
3. Average miss: model at the opener, the opener, and the close.

    python -m nbamodel.openers          # writes docs/openers.json
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import lines

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "openers.json"
MIN_GAMES = 300      # below this the result is reported but called too early to read


def _results(league, season):
    if league == "nba":
        from . import data
        g = data.games(season, refresh=True)
    else:
        from cbbmodel import data
        g = data.games(season, refresh=True)
    return g[["game_id", "margin", "total"]]


def _record(actual, line, pick_over):
    res = np.sign(actual - line) * np.where(pick_over, 1, -1)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    return {"w": w, "l": l, "p": p, "pct": round(100 * w / n, 1) if n else None,
            "roi": round(100 * (w * 100 / 110 - l) / max(1, w + l + p), 1), "se": round(100 * math.sqrt(0.25 / n), 1) if n else None}


def _side(df, model_col, open_col, close_col, actual_col, gaps):
    d = df.dropna(subset=[model_col, open_col, close_col, actual_col])
    out = {"games": int(len(d))}
    if d.empty:
        return out
    edge = d[model_col] - d[open_col]
    move = (d[close_col] - d[open_col]) * np.sign(edge)
    out["mae"] = {"model_at_open": round(float((d[actual_col] - d[model_col]).abs().mean()), 2),
                  "open": round(float((d[actual_col] - d[open_col]).abs().mean()), 2),
                  "close": round(float((d[actual_col] - d[close_col]).abs().mean()), 2)}
    out["by_gap"] = {}
    for k in gaps:
        sel = edge.abs() >= k
        sub, mv = d[sel], move[sel]
        out["by_gap"][f"{k}+"] = {
            "record_at_open": _record(sub[actual_col].to_numpy(), sub[open_col].to_numpy(), (sub[model_col] > sub[open_col]).to_numpy()),
            "line_moved_toward_model_pct": round(100 * float((mv > 0).sum()) / max(1, int((mv != 0).sum())), 1),
            "avg_move_toward_model": round(float(mv.mean()), 2) if len(mv) else None,
        }
    return out


def grade(league, season):
    log = lines.read(league)
    if log.empty:
        return {"games": 0, "note": "no lines logged yet"}
    first = log.drop_duplicates("game_id", keep="first").set_index("game_id")
    last = log.drop_duplicates("game_id", keep="last").set_index("game_id")
    d = pd.DataFrame({
        "open_spread": first.close_spread, "open_total": first.close_total,
        "model_margin": first.model_margin, "model_total": first.model_total,
        "close_spread": last.close_spread, "close_total": last.close_total,
        "n_lines": log.groupby("game_id").size(),
    }).reset_index()
    d = d.merge(_results(league, season), on="game_id", how="inner")
    out = {
        "games": int(len(d)),
        "side": _side(d, "model_margin", "open_spread", "close_spread", "margin", [0, 2, 3, 4, 6]),
        "total": _side(d, "model_total", "open_total", "close_total", "total", [0, 3, 5, 7]),
    }
    if len(d) < MIN_GAMES:
        out["note"] = f"only {len(d)} finished games with a logged line; too early to read (wait for {MIN_GAMES}+)"
    return out


def main():
    today = date.today()
    season = today.year + 1 if today.month >= 8 else today.year
    res = {"generatedAt": today.isoformat(), "season": season}
    for league in ("nba", "cbb"):
        try:
            res[league] = grade(league, season)
        except Exception as e:      # one league's data failing shouldn't stop the other
            res[league] = {"error": str(e)}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v.get("games"), v.get("note")) if isinstance(v, dict) else v for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
