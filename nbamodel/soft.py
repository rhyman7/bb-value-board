"""Softer markets: first-half spreads and totals, logged live and graded nightly.

The model has no first-half model of its own. Its first-half numbers are the full-game margin and
total scaled by how first halves have gone against the full-game model (a straight-line fit on past
seasons, saved in docs/nba/soft_params.json and refitted with the weekly backtest).

Lines come from the Odds API on the late-afternoon run (oddsapi/props.py, about 2 credits a game
on top of the props). Each logged game gets the consensus first-half line and prices, the model's
first-half numbers, and after the game the first-half result from ESPN's line scores.

    python -m nbamodel.soft fit      # refit the scaling and print how it does
    python -m nbamodel.soft grade    # grade the log into docs/nba/soft_markets.json
"""
from __future__ import annotations

import ast
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "docs" / "nba" / "soft_params.json"
LOG = ROOT / "docs" / "lines" / "nba_soft.csv"
OUT = ROOT / "docs" / "nba" / "soft_markets.json"
COLUMNS = ["game_id", "date", "fetched_at", "model_h1_margin", "model_h1_total", "h1_spread", "h1_home_price", "h1_away_price",
           "h1_spread_books", "h1_total", "h1_over_price", "h1_under_price", "h1_total_books"]
GAPS = {"h1_spread": [0, 1, 2, 3], "h1_total": [0, 1.5, 3, 4.5]}
DEFAULT = {"margin": [0.0, 0.5], "total": [0.0, 0.5]}


def half_scores(season: int) -> pd.DataFrame:
    """First-half margin (home minus away) and total from ESPN line scores in the hoopR schedule."""
    from . import data
    s = data.schedule(season, refresh=True)
    rows = []
    for r in s.itertuples():
        try:
            h = [x["value"] for x in _periods(r.home_linescores)]
            a = [x["value"] for x in _periods(r.away_linescores)]
        except Exception:
            continue
        if len(h) >= 4 and len(a) >= 4:
            rows.append({"game_id": str(r.game_id), "h1_margin": h[0] + h[1] - a[0] - a[1], "h1_total": h[0] + h[1] + a[0] + a[1]})
    return pd.DataFrame(rows, columns=["game_id", "h1_margin", "h1_total"]).astype({"h1_margin": float, "h1_total": float})


def _periods(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        raise ValueError
    if isinstance(v, str):
        v = ast.literal_eval(re.sub(r"}\s+{", "}, {", v))
    v = list(v)
    return sorted(v, key=lambda x: x["period"]) if all("period" in x for x in v) else v     # older seasons: in order, no period key


def _ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return [float(b[0]), float(b[1])]


def fit(bt: pd.DataFrame, holdout_from: int = 2025) -> dict:
    """bt: backtest games with season, m_margin, m_total. Fits on seasons before holdout_from,
    reports the holdout, then refits on everything for live use."""
    seasons = sorted(int(s) for s in bt.season.unique())
    h = pd.concat([half_scores(s) for s in seasons], ignore_index=True)
    x = bt.assign(game_id=bt.game_id.astype(str)).merge(h, on="game_id").dropna(subset=["m_margin", "m_total"])
    tr, te = x[x.season < holdout_from], x[x.season >= holdout_from]
    out = {"fitted_on": f"{seasons[0]}-{seasons[-1]}", "games": int(len(x))}
    for k in ("margin", "total"):
        b = _ols(tr[f"m_{k}"], tr[f"h1_{k}"])
        pred = b[0] + b[1] * te[f"m_{k}"]
        out[f"holdout_{k}_mae"] = {"scaled_model": round(float((te[f"h1_{k}"] - pred).abs().mean()), 2),
                                   "half_of_full": round(float((te[f"h1_{k}"] - te[f"m_{k}"] / 2).abs().mean()), 2)}
        out[k] = _ols(x[f"m_{k}"], x[f"h1_{k}"])
    PARAMS.parent.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(out, indent=1))
    return out


def params() -> dict:
    return json.loads(PARAMS.read_text()) if PARAMS.exists() else DEFAULT


def model_h1(margin, total, p=None):
    p = p or params()
    return (round(p["margin"][0] + p["margin"][1] * margin, 2) if margin is not None else None,
            round(p["total"][0] + p["total"][1] * total, 2) if total is not None else None)


def log(day: date, games: list, lines: dict) -> int:
    """games: feed games (id, date, margin, total); lines: {game_id: consensus first-half lines}."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p = params()
    rows = []
    for g in games:
        ln = lines.get(str(g["id"]))
        if not ln or g.get("date") != day.isoformat():
            continue
        hm, ht = model_h1(g.get("margin"), g.get("total"), p)
        rows.append({"game_id": str(g["id"]), "date": g["date"], "fetched_at": now, "model_h1_margin": hm, "model_h1_total": ht, **ln})
    if not rows:
        return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return len(rows)


def _payout(price):
    return np.where(price > 0, price / 100, 100 / np.abs(price))


def _record(model, line, actual, p_yes, p_no, gap):
    lean = model - line
    sel = (lean.abs() > 0) if gap == 0 else (lean.abs() >= gap)
    res = np.sign(actual - line)[sel] * np.sign(lean[sel])
    price = np.where(lean[sel] > 0, p_yes[sel], p_no[sel])
    prof = np.where(res > 0, _payout(price), np.where(res < 0, -1.0, 0.0))
    w, l, pu = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    return {"w": w, "l": l, "p": pu, "pct": round(100 * w / n, 1) if n else None,
            "roi": round(100 * float(prof.mean()), 1) if len(prof) else None, "se": round(100 * (0.25 / n) ** 0.5, 1) if n else None}


def grade(season: int) -> dict:
    if not LOG.exists():
        return {}
    lg = pd.read_csv(LOG, dtype={"game_id": str}).sort_values("fetched_at").drop_duplicates("game_id", keep="last")
    act = half_scores(season).rename(columns={"h1_margin": "act_margin", "h1_total": "act_total"})
    x = lg.merge(act, on="game_id").dropna(subset=["act_margin"])
    last = x.date.max() if len(x) else None
    out = {"generated": date.today().isoformat(), "games": int(len(x)), "last_night": last}
    for mk, mcol, acol, py, pn in (("h1_spread", "model_h1_margin", "act_margin", "h1_home_price", "h1_away_price"),
                                   ("h1_total", "model_h1_total", "act_total", "h1_over_price", "h1_under_price")):
        y = x.dropna(subset=[mcol, mk, py, pn])
        out[mk] = {"games": int(len(y)),
                   "mae": {"model": round(float((y[acol] - y[mcol]).abs().mean()), 2) if len(y) else None,
                           "line": round(float((y[acol] - y[mk]).abs().mean()), 2) if len(y) else None},
                   "by_gap": {f"{g}+": _record(y[mcol], y[mk], y[acol], y[py], y[pn], g) for g in GAPS[mk]}}
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["fit"]:
        from . import backtest
        bt = backtest.run(2026, verbose=False)["games"]
        print(json.dumps(fit(bt[bt.season >= 2021]), indent=1))
    else:
        print(json.dumps(grade(int(sys.argv[2]) if len(sys.argv) > 2 else 2027), indent=1))
