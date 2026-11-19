"""Pre-tip updates: catch injury news before the line moves.

Every 30 minutes in the hours before tip-off (GitHub Actions, .github/workflows/pretip.yml):

1. Re-read ESPN's injury report and redo the two model inputs that depend on it, missing players
   and impact-rated lineup strength, for tonight's games that haven't tipped. Everything else in
   the model's number comes from the last full update (docs/feed/nba.json).
2. Read the consensus spread from the Odds API (1 credit for every game at once), but only on the
   hourly run or when the news moved the model for one of tonight's games, which keeps it near 300
   credits a month.
3. Append a row per game to docs/lines/nba_pretip.csv: time, minutes to tip, who is out, the model's
   updated margin, the line, and whether the news moved the model.

Grading (grade(), each morning from the main update): for every time news moved the model by
NEWS_MIN points or more, the line at that moment versus the last line logged before tip. If the
model is quicker than the market, the line should keep moving toward it after the news, and the
model's side at the news-time line should win more than 52.4% against the final result.

    python -m nbamodel.pretip            # one run
    python -m nbamodel.pretip grade      # writes docs/nba/pretip.json
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import impact
from .pipeline import ET, FEED, TEAM_NAME, current_missing, injuries, upcoming

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "docs" / "lines" / "nba_pretip.csv"
OUT = ROOT / "docs" / "nba" / "pretip.json"
WINDOW_HOURS = 3.5
NEWS_MIN = 0.5          # points the model has to move for a run to count as news
COLUMNS = ["fetched_at", "game_id", "minutes_to_tip", "out_hash", "out_home", "out_away", "model_margin",
           "model_moved", "line", "books", "line_read"]


def _hash(ids):
    return hashlib.md5(",".join(str(i) for i in sorted(ids)).encode()).hexdigest()[:10]


def odds_spreads():
    """{(home, away): consensus home spread, our sign (+ = home favored)} for every NBA game posted."""
    from oddsapi import client
    js, _ = client.get(f"/sports/{client.SPORTS['nba']}/odds", regions="us", markets="spreads", oddsFormat="american")
    out = {}
    for e in js:
        pts = []
        for bk in e.get("bookmakers", []):
            for m in bk.get("markets", []):
                for o in m.get("outcomes", []):
                    if o.get("name") == e["home_team"] and o.get("point") is not None:
                        pts.append(-float(o["point"]))
        if pts:
            out[(e["home_team"], e["away_team"])] = (statistics.median(pts), len(pts))
    return out


def run(now=None):
    now = now or datetime.now(ET)
    season = now.year + 1 if now.month >= 8 else now.year
    feed = json.loads(FEED.read_text())
    by_id = {g["id"]: g for g in feed.get("games", [])}
    coef = feed.get("coefficients", {})
    sched = upcoming(season, now.date())
    soon = sched[(sched.et > now) & (sched.et <= now + timedelta(hours=WINDOW_HOURS))]
    soon = soon[soon.game_id.astype(str).isin(by_id)]
    if soon.empty:
        print("no games tipping in the next", WINDOW_HOURS, "hours")
        return 0
    inj = injuries()
    if not inj:     # in season ESPN always lists someone; an empty report is a failed read
        print("injury report not read or empty; nothing logged")
        return 0
    miss, names, _ = current_missing(None, season, inj)
    try:
        strength, _, _ = impact.today(season, set(inj))
    except Exception as e:
        print("impact ratings failed:", repr(e))
        strength = {}
    out_hash = _hash(inj)
    prev = pd.read_csv(LOG, dtype={"game_id": str}) if LOG.exists() else pd.DataFrame(columns=COLUMNS)
    last_hash = prev.out_hash.iloc[-1] if len(prev) else None
    news = out_hash != last_hash
    rows = []
    for x in soon.itertuples():
        g = by_id[str(x.game_id)]
        hid, aid = int(x.home_id), int(x.away_id)
        has_mk = g.get("market_ratings_margin") is not None
        cm = coef.get("margin" if has_mk else "margin_base", coef.get("margin", {}))
        old_miss = (g.get("missing") or {}).get("home", 0.0) - (g.get("missing") or {}).get("away", 0.0)
        new_miss = miss.get(hid, 0.0) - miss.get(aid, 0.0)
        imp = g.get("imp") or {}
        old_imp = (imp.get("home") - imp.get("away")) if imp.get("home") is not None and imp.get("away") is not None else None
        new_imp = (strength[hid] - strength[aid]) if hid in strength and aid in strength else None
        m = g["margin"] + cm.get("miss_diff", 0.0) * (new_miss - old_miss)
        if old_imp is not None and new_imp is not None:
            m += cm.get("imp_diff", 0.0) * (new_imp - old_imp)
        last = prev[prev.game_id == str(x.game_id)].model_margin
        moved = float(m - (last.iloc[-1] if len(last) else g["margin"]))
        rows.append({"_teams": (TEAM_NAME.get(x.home_display_name, x.home_display_name), TEAM_NAME.get(x.away_display_name, x.away_display_name)),
                     "fetched_at": now.astimezone(timezone.utc).isoformat(timespec="seconds"), "game_id": str(x.game_id),
                     "minutes_to_tip": int((x.et - now).total_seconds() // 60), "out_hash": out_hash,
                     "out_home": "; ".join(names.get(hid, [])), "out_away": "; ".join(names.get(aid, [])),
                     "model_margin": round(m, 2), "model_moved": round(moved, 2),
                     "line": None, "books": None, "line_read": False})
    # the line: on the hourly run, or when the news moved the model for a game tonight
    hit = any(abs(r["model_moved"]) >= 0.1 for r in rows)
    lines = {}
    if os.environ.get("ODDS_API_KEY") and (hit or now.minute < 30):
        try:
            lines = odds_spreads()
        except Exception as e:
            print("odds failed:", repr(e))
    for r in rows:
        ln = lines.get(r.pop("_teams"))
        r.update({"line": ln[0] if ln else None, "books": ln[1] if ln else None, "line_read": bool(lines)})
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"{len(rows)} games logged; injury report {'changed' if news else 'unchanged'}; model moved {'yes' if hit else 'no'}; lines {'read' if lines else 'not read'}")
    return len(rows)


def _record(model, line, actual):
    res = np.sign(actual - line) * np.sign(model - line)
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    n = w + l
    return {"w": w, "l": l, "p": p, "pct": round(100 * w / n, 1) if n else None, "se": round(100 * (0.25 / n) ** 0.5, 1) if n else None}


def grade(season=None):
    if not LOG.exists():
        return {}
    from . import data
    now = datetime.now(ET)
    season = season or (now.year + 1 if now.month >= 8 else now.year)
    lg = pd.read_csv(LOG, dtype={"game_id": str}).sort_values("fetched_at")
    res = data.games(season)[["game_id", "margin"]].assign(game_id=lambda d: d.game_id.astype(str))
    withline = lg.dropna(subset=["line"])
    close = withline.groupby("game_id").line.last().rename("close")
    # news: a run where the model moved NEWS_MIN+ and the line was read in that same run
    ev = withline[withline.model_moved.abs() >= NEWS_MIN].merge(close, on="game_id").merge(res, on="game_id")
    ev = ev.drop_duplicates("game_id", keep="first")        # the first news of each game
    lean = ev.model_margin - ev.line
    mv = ev.close - ev.line
    moved = (mv != 0) & (lean != 0)
    out = {"generated": now.date().isoformat(), "runs": int(lg.fetched_at.nunique()), "games_logged": int(lg.game_id.nunique()),
           "news_events": int(len(ev)),
           "line_moved_toward_model_pct": round(100 * float((np.sign(mv[moved]) == np.sign(lean[moved])).mean()), 1) if moved.any() else None,
           "avg_move_toward_model": round(float((mv * np.sign(lean))[lean != 0].mean()), 2) if (lean != 0).any() else None,
           "record_at_news_line": _record(ev.model_margin, ev.line, ev.margin),
           "record_at_close_same_side": _record(ev.model_margin, ev.close, ev.margin)}
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["grade"]:
        print(json.dumps(grade(), indent=1))
    else:
        run()
