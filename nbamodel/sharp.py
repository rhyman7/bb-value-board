"""Sharp-price spreads: a US-regulated book's NBA spread that is off Pinnacle's fair price.

Pinnacle is the sharpest book. Its two prices give a no-vig chance that the home side covers at its
line, which sets the market's expected margin m (normal, sd SD points). That prices any other line:
P(home covers at spread s) = Phi((m + s) / SD). A flag is a regulated book whose line and price beat
that by MIN_EV or more.

On past lines (experiments/sharp_price.py, 2024-26, 11 AM reads): 145 of 2,195 games had a
positive-value bet; they won 55.9%, +9.4% +/- 7.9%, and beat the closing line 54% of the time
(59% at 2%+). Small, so tracked here before anything is trusted.

The pre-tip loop (pretip.py) reads one call per run with Pinnacle and nine US books named (the
same 1 credit as a whole region), logs flags to docs/lines/nba_sharp.csv, and grade() writes
docs/nba/sharp.json: tonight's flags and the record (first flag of each game), with closing-line value.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "docs" / "lines" / "nba_sharp.csv"
OUT = ROOT / "docs" / "nba" / "sharp.json"
BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers", "fanatics", "espnbet",
         "hardrockbet", "ballybet"]
NAMES = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
         "betrivers": "BetRivers", "fanatics": "Fanatics", "espnbet": "ESPN BET", "hardrockbet": "Hard Rock",
         "ballybet": "Bally Bet"}
SD = 12.5
MIN_EV = 0.02
N = NormalDist()
COLS = ["fetched_at", "game_id", "home", "away", "minutes_to_tip", "side", "team", "spread", "price", "book",
        "pin_spread", "pin_home_price", "pin_away_price", "fair", "ev"]


def _dec(p):
    return 1 + (p / 100 if p > 0 else 100 / -p)


def fetch():
    from oddsapi import client
    js, _ = client.get(f"/sports/{client.SPORTS['nba']}/odds", bookmakers=",".join(BOOKS), markets="spreads",
                       oddsFormat="american")
    return js


def _quotes(e):
    """[(book, home spread (Odds API sign: - = home favored), home price, away price)]."""
    out = []
    for bk in e.get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != "spreads":
                continue
            o = {x["name"]: x for x in m.get("outcomes", []) if x.get("point") is not None}
            h, a = o.get(e["home_team"]), o.get(e["away_team"])
            if h and a:
                out.append((bk["key"], float(h["point"]), int(h["price"]), int(a["price"])))
    return out


def consensus(js):
    """{(home, away): (median home spread in our sign, + = home favored; books)} like pretip.odds_spreads."""
    import statistics
    out = {}
    for e in js:
        q = _quotes(e)
        if q:
            out[(e["home_team"], e["away_team"])] = (statistics.median(-x[1] for x in q), len(q))
    return out


def flags(js, games, now_iso):
    """games: {(home, away): (game_id, minutes_to_tip)}. Returns rows for every regulated quote with
    EV >= MIN_EV against Pinnacle's fair price."""
    rows = []
    for e in js:
        key = (e["home_team"], e["away_team"])
        if key not in games:
            continue
        q = _quotes(e)
        pin = [x for x in q if x[0] == "pinnacle"]
        if not pin:
            continue
        _, ps, ph, pa = pin[0]
        a, b = 1 / _dec(ph), 1 / _dec(pa)
        fair_home = min(max(a / (a + b), 1e-4), 1 - 1e-4)
        m = SD * N.inv_cdf(fair_home) - ps
        gid, mins = games[key]
        for book, s, hp, ap in q:
            if book == "pinnacle":
                continue
            for side, team, price, p in (("home", e["home_team"], hp, N.cdf((m + s) / SD)),
                                         ("away", e["away_team"], ap, 1 - N.cdf((m + s) / SD))):
                ev = p * _dec(price) - 1
                if ev >= MIN_EV:
                    rows.append({"fetched_at": now_iso, "game_id": gid, "home": e["home_team"], "away": e["away_team"],
                                 "minutes_to_tip": mins, "side": side, "team": team,
                                 "spread": s if side == "home" else -s, "price": price, "book": NAMES.get(book, book),
                                 "pin_spread": ps, "pin_home_price": ph, "pin_away_price": pa,
                                 "fair": round(p, 3), "ev": round(ev, 3)})
    return rows


def log(rows):
    if not rows:
        return 0
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return len(rows)


def best(rows):
    """The best quote for each game and side, highest EV first."""
    rows = sorted(rows, key=lambda r: -r["ev"])
    seen, out = set(), []
    for r in rows:
        k = (r["game_id"], r["side"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def update_tonight(rows, now_iso):
    """Every pre-tip run: this run's flags (possibly none) replace tonight's list in sharp.json."""
    out = json.loads(OUT.read_text()) if OUT.exists() else {"record": None}
    out.update({"read_at": now_iso, "tonight": best(rows)})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))


def grade(season, pretip_log):
    """First flag of each game (its best quote at that read), graded on the final margin and against
    the last consensus line logged before tip."""
    from . import data
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    out = {"generated": datetime.now().date().isoformat(), "read_at": prev.get("read_at"),
           "tonight": prev.get("tonight", []), "record": None,
           "backtest": "Past lines, 2024-26: 145 bets, 55.9%, +9.4% ± 7.9%; beat the closing line 54% of the time."}
    if not LOG.exists():
        OUT.write_text(json.dumps(out, indent=1))
        return out
    lg = pd.read_csv(LOG, dtype={"game_id": str}).sort_values(["fetched_at", "ev"], ascending=[True, False])
    first = lg.drop_duplicates("game_id", keep="first").copy()
    res = data.games(season)[["game_id", "margin"]].assign(game_id=lambda d: d.game_id.astype(str))
    x = first.merge(res, on="game_id")
    if len(x):
        cover = x.margin + np.where(x.side == "home", x.spread, -x.spread)       # home margin + home handicap
        r = np.where(x.side == "home", np.sign(cover), -np.sign(cover))
        x["profit"] = np.where(r > 0, x.price.map(_dec) - 1, np.where(r < 0, -1.0, 0.0))
        x["res"] = r
        rec = {"bets": int(len(x)), "w": int((r > 0).sum()), "l": int((r < 0).sum()), "p": int((r == 0).sum()),
               "units": round(float(x.profit.sum()), 2), "roi": round(100 * float(x.profit.mean()), 1)}
        if pretip_log is not None and len(pretip_log):
            close = pretip_log.dropna(subset=["line"]).sort_values("fetched_at").groupby("game_id").line.last()   # + = home favored
            c = x.join(close.rename("close"), on="game_id").dropna(subset=["close"])
            if len(c):
                # points gained on the bettor's handicap: the close is + = home favored, so the home
                # handicap at the close is -close and the away handicap is +close
                clv = np.where(c.side == "home", c.spread + c.close, c.spread - c.close)
                rec["avg_clv_pts"] = round(float(np.mean(clv)), 2)
                rec["beat_close_pct"] = round(100 * float(np.mean(clv > 0)), 1)
        out["record"] = rec
    OUT.write_text(json.dumps(out, indent=1, default=float))
    return out
