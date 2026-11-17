"""Would the models make money betting moneylines?

hoopR keeps closing spreads but no moneylines, so the market's win probability here comes from the
closing spread: for each season, a logistic fit of the home team's win on the closing spread, using
earlier seasons only. That is close to the no-vig moneyline (books price the two together).
The model's win probability gets the same treatment from its own margin. Bets: whenever the model's
probability beats the market's by EDGE or more on a side, one unit at the market price with a
standard 4.5% hold split across the two sides (and at the no-vig price, as an upper bound).

    python -m experiments.moneyline      # writes docs/moneyline.json
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HOLD = 0.045
EDGES = [0.0, 0.02, 0.04, 0.06, 0.08]
BUCKETS = [(0.0, 0.25, "Long underdog (under 25%)"), (0.25, 0.5, "Underdog (25-50%)"),
           (0.5, 0.75, "Favorite (50-75%)"), (0.75, 1.01, "Heavy favorite (over 75%)")]
OUT = Path(__file__).resolve().parent.parent / "docs" / "moneyline.json"


def logit_fit(x, y):
    """Slope-only logistic fit (no intercept: home court is already in both numbers)."""
    b = 0.15
    for _ in range(50):
        p = 1 / (1 + np.exp(-b * x))
        g = np.sum((y - p) * x)
        h = -np.sum(p * (1 - p) * x * x)
        b -= g / h
    return b


def probs(df, col, first_graded):
    out = pd.Series(np.nan, index=df.index)
    for s in sorted(df.season.unique()):
        if s < first_graded:
            continue
        tr = df[(df.season < s)]
        b = logit_fit(tr[col].to_numpy(float), (tr.margin > 0).to_numpy(float))
        sel = df.season == s
        out[sel] = 1 / (1 + np.exp(-b * df.loc[sel, col].to_numpy(float)))
    return out


def bets(d, edge, hold):
    rows = []
    for side in ("h", "a"):
        pm = d.p_model if side == "h" else 1 - d.p_model
        pk = d.p_mkt if side == "h" else 1 - d.p_mkt
        won = (d.margin > 0) if side == "h" else (d.margin < 0)
        sel = (pm - pk) >= edge if edge > 0 else (pm > pk)
        price = 1 / np.clip(pk + hold / 2, 0.01, 0.995)       # decimal odds
        rows.append(pd.DataFrame({"pk": pk[sel], "won": won[sel], "price": price[sel]}))
    b = pd.concat(rows)
    return b


def summarize(b):
    n = len(b)
    if not n:
        return {"bets": 0}
    profit = np.where(b.won, b.price - 1, -1.0)
    roi = profit.mean()
    se = profit.std(ddof=1) / math.sqrt(n) if n > 1 else None
    return {"bets": n, "won": int(b.won.sum()), "win_pct": round(100 * b.won.mean(), 1),
            "avg_price_american": american(float(np.median(b.price))), "roi_pct": round(100 * roi, 1),
            "roi_se": round(100 * se, 1) if se else None}


def american(dec):
    return round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1))


def scores(d):
    y = (d.margin > 0).astype(float)
    out = {}
    for k in ["p_model", "p_mkt"]:
        p = d[k].clip(1e-4, 1 - 1e-4)
        out[k] = {"brier": round(float(((p - y) ** 2).mean()), 4), "log_loss": round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 4)}
    return out


def league(d, first_graded, seasons):
    d = d.dropna(subset=["close_spread", "m_margin"]).copy()
    d = d[d.margin != 0]
    d["p_model"] = probs(d, "m_margin", first_graded)
    d["p_mkt"] = probs(d, "close_spread", first_graded)
    g = d[d.season.isin(seasons)].dropna(subset=["p_model", "p_mkt"])
    res = {"seasons": f"{min(seasons)}-{max(seasons)}", "games": int(len(g)), "accuracy": scores(g), "by_edge": {}, "by_price": {}}
    for e in EDGES:
        res["by_edge"][f"{int(e * 100)}%+"] = {"with_hold": summarize(bets(g, e, HOLD)), "no_vig": summarize(bets(g, e, 0.0))}
    b = bets(g, 0.04, HOLD)
    for lo, hi, lab in BUCKETS:
        res["by_price"][lab] = summarize(b[(b.pk >= lo) & (b.pk < hi)])
    return res


def main():
    from nbamodel import backtest as nbt
    from cbbmodel import backtest as cbt
    nba = nbt.run(2027, verbose=False)["games"]
    nba = nba[nba.season < 2027]
    cbb = cbt.run(2027, verbose=False)["games"]
    cbb = cbb[~cbb.neutral]
    out = {"note": "Market win probability from the closing spread (hoopR has no moneylines); 4.5% hold.",
           "nba": league(nba, 2022, list(range(2022, 2027))),
           "cbb": league(cbb, 2019, cbt.GRADE)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
