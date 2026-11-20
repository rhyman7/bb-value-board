"""Props, round two: (1) the best price instead of the typical one, (2) betting only when the
model's chance beats the price.

Uses the 100 games of past props already pulled (data/odds/nba_props.csv.gz, every book 60 minutes
before tip). Chances come from nbamodel/propprob.py with k fitted on 2021-23, before any of these
lines. Books: "legal" = US-regulated books (offshore betonlineag, bovada and mybookieag left out).

    python -m experiments.props_value       # writes docs/nba/prop_dist.json and docs/nba/props_value.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nbamodel import players as P, propprob
from oddsapi.prop_history import PATH
from oddsapi.props import MARKETS, _norm
from experiments.props_backtest import decimal

OUT = Path(__file__).resolve().parent.parent / "docs" / "nba" / "props_value.json"
OFFSHORE = {"betonlineag", "bovada", "mybookieag"}
THRESH = (0.0, 0.03, 0.06, 0.10)


def quotes():
    q = pd.read_csv(PATH).dropna(subset=["player", "line"])
    q["stat"] = q.market.map(MARKETS)
    q["date"] = pd.to_datetime(q.commence_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    q["who"] = q.player.map(_norm)
    q["over"], q["under"] = decimal(q.over), decimal(q.under)
    q["legal"] = ~q.book.isin(OFFSHORE)
    return q


def summ(d):
    if d.empty:
        return {"n": 0}
    n = len(d)
    w = d.ret > 0
    return {"n": int(n), "win": round(float(w.mean()), 4), "roi": round(float(d.ret.mean()), 4),
            "roi_se": round(float(d.ret.std() / np.sqrt(n)), 4)}


def result(side_over, line, actual, dec):
    push = actual == line
    won = np.where(side_over, actual > line, actual < line)
    return np.where(push, 0.0, np.where(won, dec - 1.0, -1.0))


def novig_over(o, u):
    a, b = 1 / o, 1 / u
    return a / (a + b)


def blend(c):
    """The model alone is overconfident (a 62% call wins 53%). Blend its chance with the market's
    no-vig chance (from the median prices): p = w * model + (1 - w) * market. w is picked on 2023-24
    by log loss, then value bets at the best legal price are checked on 2024-25 and 2025-26."""
    d = c.dropna(subset=["legal_over", "legal_under"]).copy()
    d = d[d.actual != d.line]
    mk = novig_over(d.med_over, d.med_under)
    nopush = 1 - d.p_push
    model = d.p_over / nopush                       # chance of over given no push
    y = (d.actual > d.line).astype(float)
    fit = d.season == 2024
    best, out = None, {}
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        pr = np.clip(w * model + (1 - w) * mk, 1e-4, 1 - 1e-4)
        ll = float(-(y * np.log(pr) + (1 - y) * np.log(1 - pr))[fit].mean())
        out[str(w)] = round(ll, 5)
        if best is None or ll < best[0]:
            best = (ll, w)
    w = best[1]
    pr = w * model + (1 - w) * mk
    ll = lambda q, m: float(-(y * np.log(q) + (1 - y) * np.log(1 - q))[m].mean())
    test = d.season >= 2025
    r = {"w": float(w), "logloss_fit_by_w": out,
         "test_logloss": {"market": round(ll(np.clip(mk, 1e-4, 1 - 1e-4), test), 5),
                          "model": round(ll(np.clip(model, 1e-4, 1 - 1e-4), test), 5),
                          "blend": round(ll(pr, test), 5)}}
    ev_o = pr * (d.legal_over - 1) - (1 - pr)
    ev_u = (1 - pr) * (d.legal_under - 1) - pr
    side = ev_o >= ev_u
    ev = np.maximum(ev_o, ev_u)
    ret = result(side, d.line, d.actual, np.where(side, d.legal_over, d.legal_under))
    t = pd.DataFrame({"ev": ev, "ret": ret, "season": d.season, "stat": d.stat})
    r["value_bets_test_2025_26"] = {f"edge {int(x * 100)}%+": {"all": summ(t[test & (t.ev >= x)]),
                                                              "by_stat": {s: summ(g) for s, g in t[test & (t.ev >= x)].groupby("stat")}}
                                    for x in (0.0, 0.02, 0.04, 0.06)}
    r["value_bets_fit_2024"] = {f"edge {int(x * 100)}%+": summ(t[fit & (t.ev >= x)]) for x in (0.0, 0.02, 0.04, 0.06)}
    return r


def main():
    k = {}
    walk = P.walk([2020, 2021, 2022, 2023, 2024, 2025, 2026])
    params = P.load_params()
    proj = P.project(walk, params)
    proj = proj[proj.a_minutes.notna()]
    tune = P.graded(proj[proj.season.isin(P.TUNE)])
    for s in P.STATS:
        k[s] = round(propprob.fit_k(tune[f"p_{s}"], tune[f"a_{s}"]), 4)
    propprob.PARAMS.write_text(json.dumps({"k": k, "fit_on": "2021-23 player-games, projections vs actual",
                                           "n": int(len(tune))}, indent=1))

    p = proj[proj.season >= 2024].copy()
    p["date"] = pd.to_datetime(p.date).dt.strftime("%Y-%m-%d")
    p["who"] = p.name.map(_norm)
    long = pd.concat([p[["date", "who", "season", f"p_{s}", f"a_{s}"]].rename(columns={f"p_{s}": "proj", f"a_{s}": "actual"})
                      .assign(stat=s) for s in P.STATS])
    q = quotes().merge(long, on=["date", "who", "stat"], how="inner")
    # chances at every quoted line
    q["p_over"] = q["p_under"] = q["p_push"] = np.nan
    for s in P.STATS:
        m = q.stat == s
        po, pu, pp = propprob.chances(q.loc[m, "proj"].to_numpy(), q.loc[m, "line"].to_numpy(), k[s])
        q.loc[m, "p_over"], q.loc[m, "p_under"], q.loc[m, "p_push"] = po, pu, pp
    q["ev_over"] = propprob.edge(q.p_over, q.p_push, q.over)
    q["ev_under"] = propprob.edge(q.p_under, q.p_push, q.under)

    key = ["date", "who", "stat"]
    cons_line = q.groupby(key).line.median().rename("cons_line")
    q = q.join(cons_line, on=key)
    at = q[q.line == q.cons_line]
    rows = []
    for (date, who, stat), g in at.groupby(key):
        r = g.iloc[0]
        lg = g[g.legal]
        rows.append({"date": date, "who": who, "stat": stat, "season": r.season, "line": r.line, "proj": r.proj,
                     "actual": r.actual, "p_over": r.p_over, "p_under": r.p_under, "p_push": r.p_push,
                     "med_over": g.over.median(), "med_under": g.under.median(),
                     "best_over": g.over.max(), "best_under": g.under.max(),
                     "legal_over": lg.over.max() if len(lg) else np.nan, "legal_under": lg.under.max() if len(lg) else np.nan,
                     "books": len(g)})
    c = pd.DataFrame(rows)
    side = c.proj > c.line
    c = c[c.proj != c.line]
    side = c.proj > c.line
    res = {"k": k, "props": int(len(c))}

    def lean(price_over, price_under, d=c, s=None):
        s = (d.proj > d.line) if s is None else s
        dec = np.where(s, d[price_over], d[price_under])
        ok = ~np.isnan(dec)
        return pd.DataFrame({"ret": result(s[ok], d.line[ok], d.actual[ok], dec[ok]), "stat": d.stat[ok].values,
                             "season": d.season[ok].values})
    res["lean_side_median_price"] = summ(lean("med_over", "med_under"))
    res["lean_side_best_price_any_book"] = summ(lean("best_over", "best_under"))
    res["lean_side_best_price_legal"] = summ(lean("legal_over", "legal_under"))
    res["lean_side_best_legal_by_stat"] = {s: summ(g) for s, g in lean("legal_over", "legal_under").groupby("stat")}

    # value bets: the side whose chance beats the best legal price at the consensus line
    d = c.dropna(subset=["legal_over", "legal_under"]).copy()
    d["ev_o"] = propprob.edge(d.p_over, d.p_push, d.legal_over)
    d["ev_u"] = propprob.edge(d.p_under, d.p_push, d.legal_under)
    d["side_over"] = d.ev_o >= d.ev_u
    d["ev"] = np.where(d.side_over, d.ev_o, d.ev_u)
    d["dec"] = np.where(d.side_over, d.legal_over, d.legal_under)
    d["ret"] = result(d.side_over, d.line, d.actual, d.dec)
    d["p_win"] = np.where(d.side_over, d.p_over, d.p_under)
    res["value_bets_best_legal"] = {f"edge {int(t * 100)}%+": {"all": summ(d[d.ev >= t]),
                                                             "by_stat": {s: summ(g) for s, g in d[d.ev >= t].groupby("stat")},
                                                             "by_season": {int(y): summ(g) for y, g in d[d.ev >= t].groupby("season")}}
                                    for t in THRESH}
    # same at the median price (what a single typical book offers)
    d2 = c.copy()
    d2["ev_o"] = propprob.edge(d2.p_over, d2.p_push, d2.med_over)
    d2["ev_u"] = propprob.edge(d2.p_under, d2.p_push, d2.med_under)
    d2["side_over"] = d2.ev_o >= d2.ev_u
    d2["ev"] = np.maximum(d2.ev_o, d2.ev_u)
    d2["ret"] = result(d2.side_over, d2.line, d2.actual, np.where(d2.side_over, d2.med_over, d2.med_under))
    res["value_bets_median_price"] = {f"edge {int(t * 100)}%+": summ(d2[d2.ev >= t]) for t in THRESH}
    # calibration: does a 60% chance win 60% of the time?
    d["bucket"] = pd.cut(d.p_win, [0, 0.45, 0.5, 0.55, 0.6, 0.65, 1.0])
    res["calibration"] = [{"chance": str(b), "n": int(len(g)), "model_avg": round(float(g.p_win.mean()), 3),
                           "won": round(float((g.ret > 0).mean()), 3)} for b, g in d.groupby("bucket", observed=True)]
    res["blend"] = blend(c)
    prm = json.loads(propprob.PARAMS.read_text())
    prm.update({"w": res["blend"]["w"], "min_edge": 0.03,
                "w_note": "model weight in the chance; fit on 2023-24 past props by log loss (experiments/props_value.py)"})
    propprob.PARAMS.write_text(json.dumps(prm, indent=1))
    print(json.dumps(res, indent=1))
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

