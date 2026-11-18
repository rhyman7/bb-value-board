"""Walk-forward backtest of the NBA game model, graded against consensus closing lines.

Each season's game model is fitted on earlier seasons only, and every game's ratings use only
games before its date. Closing lines exist from the 2020 restart on, so the betting record and the
market weight start with the 2021 season.

    python -m nbamodel.backtest            # writes docs/nba/backtest.json and prints the summary
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import availability, data, impact, market, model, ratings

FIRST_FIT = 2017          # the game model needs at least one earlier season of rated games
GRADE_FROM = 2021         # first season with closing lines for the whole season
OUT = Path(__file__).resolve().parent.parent / "docs" / "nba"


def ats(side_margin, line):
    """Win/loss/push for betting the home side at `line` (+ = home favored) given the actual margin."""
    d = side_margin - line
    return np.sign(d)


def record(df: pd.DataFrame, pick_home: np.ndarray, actual, line):
    res = np.sign(actual - line) * np.where(pick_home, 1, -1)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    roi = (w * (100 / 110) - l) / (w + l + p) if (w + l + p) else 0.0
    return {"w": w, "l": l, "p": p, "pct": round(100 * w / n, 1) if n else None, "roi": round(100 * roi, 1),
            "se": round(100 * math.sqrt(0.25 / n), 1) if n else None}


def impact_for(seasons):
    """imp_diff for every game in `seasons`. Finished seasons come from the cache; a season in
    progress (today's) is walked from fresh box scores."""
    from datetime import date
    d = date.today()
    now = d.year + 1 if d.month >= 8 else d.year
    done = [s for s in seasons if s < now]
    parts = [impact.history(done)] if done else []
    if now in seasons:
        f, ix = impact.walk([now], current=now)
        parts.append(impact.game_strength([now], f, ix, current=now))
    return pd.concat(parts, ignore_index=True)


def run(last_season: int, settings: ratings.Settings | None = None, verbose: bool = True, use_availability: bool = True,
        extra: pd.DataFrame | None = None) -> dict:
    """extra: optional per-game feature columns (with game_id) merged in before the model, for experiments."""
    g = data.all_games(last_season)
    seasons = [s for s in sorted(g.season.unique()) if s > g.season.min()]
    rated = ratings.rating_projection(ratings.walk_forward(g, settings, seasons=seasons))
    if use_availability:
        miss = availability.missing_value(g, seasons)
        rated = rated.merge(miss, on="game_id", how="left")
        # lineup strength from impact ratings: like missing players, it knows who played
        rated = rated.merge(impact_for(seasons)[["game_id", "imp_diff"]], on="game_id", how="left")
    rated = rated.merge(data.closing_lines(), on="game_id", how="left")
    if use_availability:
        rated = rated.merge(market.market_ratings(rated, [s for s in seasons if s >= GRADE_FROM]), on="game_id", how="left")
    if extra is not None:
        rated = rated.merge(extra, on="game_id", how="left")
        cols = [c for c in extra.columns if c != "game_id"]
        rated[cols] = rated[cols].fillna(0.0)      # a game the experiment has no value for counts as neutral
    rated = model.features(rated)

    out = []
    coefs = {}
    for s in [x for x in seasons if x >= FIRST_FIT]:
        train = rated[rated.season < s]
        coef = model.fit(train)
        coefs[s] = coef
        out.append(model.predict(rated[rated.season == s], coef))
    bt = pd.concat(out, ignore_index=True)

    # market weight for each graded season, measured on earlier graded seasons only
    bt["w_side"] = np.nan
    bt["w_total"] = np.nan
    for s in sorted(bt.season.unique()):
        prev = bt[(bt.season < s) & (bt.season >= GRADE_FROM - 1)]
        ws, _, _ = model.market_weight(prev, "m_margin", "close_spread", "margin")
        wt, _, _ = model.market_weight(prev, "m_total", "close_total", "total")
        bt.loc[bt.season == s, "w_side"] = min(max(ws, 0.0), 1.0)
        bt.loc[bt.season == s, "w_total"] = min(max(wt, 0.0), 0.3)

    done = bt[bt.season < last_season]          # finished seasons only
    graded = done[(done.season >= GRADE_FROM) & done.close_spread.notna()]
    late = graded[graded.n_this >= 150]

    def mae(a, b):
        return round(float((a - b).abs().mean()), 2)

    summary = {
        "seasons": f"{int(graded.season.min())}-{int(graded.season.max())}",
        "games": int(len(graded)),
        "margin_mae": {"model": mae(graded.margin, graded.m_margin), "ratings_only": mae(graded.margin, graded.r_margin),
                       "market_ratings_only": mae(graded.margin, graded.mk_margin),
                       "close": mae(graded.margin, graded.close_spread)},
        "total_mae": {"model": mae(graded.total, graded.m_total), "close": mae(graded.total, graded.close_total)},
        "after_150_games": {"games": int(len(late)), "model": mae(late.margin, late.m_margin), "close": mae(late.margin, late.close_spread)},
        "winner_pct": {
            "model": round(100 * float(((graded.m_margin > 0) == (graded.margin > 0)).mean()), 1),
            "close": round(100 * float(((graded.close_spread > 0) == (graded.margin > 0)).mean()), 1),
        },
    }
    # one bet a game at the close, on the model's side
    pick = (graded.m_margin > graded.close_spread).to_numpy()
    summary["ats_every_game"] = record(graded, pick, graded.margin.to_numpy(), graded.close_spread.to_numpy())
    buckets = {}
    for k in [1, 2, 3, 4]:
        sel = (graded.m_margin - graded.close_spread).abs() >= k
        sub = graded[sel]
        buckets[f"{k}+"] = record(sub, (sub.m_margin > sub.close_spread).to_numpy(), sub.margin.to_numpy(), sub.close_spread.to_numpy())
    summary["ats_by_gap"] = buckets
    tpick = (graded.m_total > graded.close_total).to_numpy()
    summary["totals_every_game"] = record(graded, tpick, graded.total.to_numpy(), graded.close_total.to_numpy())
    tb = {}
    for k in [3, 5, 7]:
        sub = graded[(graded.m_total - graded.close_total).abs() >= k]
        tb[f"{k}+"] = record(sub, (sub.m_total > sub.close_total).to_numpy(), sub.total.to_numpy(), sub.close_total.to_numpy())
    summary["totals_by_gap"] = tb
    ws, se, n = model.market_weight(graded, "m_margin", "close_spread", "margin")
    wt, set_, nt = model.market_weight(graded, "m_total", "close_total", "total")
    summary["market_weight"] = {"side": round(ws, 3), "side_se": round(se, 3) if se else None,
                                "total": round(wt, 3), "total_se": round(set_, 3) if set_ else None}
    by_season = []
    for s, x in graded.groupby("season"):
        by_season.append({"season": int(s), "games": int(len(x)), "model": mae(x.margin, x.m_margin), "close": mae(x.margin, x.close_spread),
                          "ats": record(x, (x.m_margin > x.close_spread).to_numpy(), x.margin.to_numpy(), x.close_spread.to_numpy())})
    summary["by_season"] = by_season
    summary["coefficients"] = {str(k): {kk: {a: round(b, 3) for a, b in vv.items()} for kk, vv in v.items()} for k, v in coefs.items()}
    if verbose:
        print(json.dumps({k: v for k, v in summary.items() if k != "coefficients"}, indent=1))
    return {"summary": summary, "games": bt, "rated": rated, "coefs": coefs}


def main():
    from datetime import date
    today = date.today()
    last = today.year + 1 if today.month >= 8 else today.year
    res = run(last)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "backtest.json").write_text(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
