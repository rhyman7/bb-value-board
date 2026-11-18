"""What a game page needs beyond the projection: how the projection is built, each team's four
factors with league ranks, and each team's last ten games.

    build(row, coef, miss_detail)   -> {"margin": [...], "total": [...]}, each piece in points
    team_stats(tb)                  -> {abbr: {"off": {...}, "def": {...}, "ranks": {...}}}
    last_ten(games, spreads)        -> {abbr: [{date, opp, site, score, result, ats}, ...]}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = {   # key: (label, higher is better for the offense)
    "efg": ("Effective FG%", True), "tov": ("Turnover %", False), "orb": ("Off. rebound %", True),
    "ftr": ("FT rate (FTM/FGA)", True), "tpr": ("3-point rate (3PA/FGA)", None),
    "ortg": ("Points per 100", True), "pace": ("Pace", None),
}


def _piece(label, pts, kind):
    return {"label": label, "pts": round(float(pts), 2), "kind": kind}


def build(row, coef: dict, home: str, away: str, miss_detail: dict, has_mk: bool) -> dict:
    """Each input of the game model in points, adding up to the model's margin and total.
    `row` is the predicted row (ratings, features); `miss_detail` maps 'h'/'a' to [(name, value)]."""
    cm = coef["margin"] if has_mk else coef.get("margin_base", coef["margin"])
    ct = coef["total"] if has_mk else coef.get("total_base", coef["total"])
    pace = float(row.r_pace)
    hca = 0.0 if bool(row.neutral) else float(row.lg_hca)
    a = cm.get("r_margin", 0.0)
    net_h = float(row.off_h + row.def_h)
    net_a = float(row.off_a + row.def_a)
    m = [_piece(f"{home} team rating ({net_h:+.1f} per 100)", a * pace * net_h / 100, "team"),
         _piece(f"{away} team rating ({net_a:+.1f} per 100)", -a * pace * net_a / 100, "team"),
         _piece("Home court" if hca else "Neutral floor", a * pace * hca / 100, "site")]
    if has_mk and "mk_margin" in cm:
        m.append(_piece("Market's view of the teams (earlier closing lines)", cm["mk_margin"] * float(row.mk_margin), "market"))
    for k, lab in (("rest_diff", "Rest days"), ("b2b_h", f"{home} on a back-to-back"), ("b2b_a", f"{away} on a back-to-back"),
                   ("g7_diff", "Games in the last 7 days"), ("far_road_diff", "Long road trip (2+ time zones from home)")):
        v = cm.get(k, 0.0) * float(getattr(row, k, 0.0))
        if abs(v) >= 0.005:
            m.append(_piece(lab, v, "schedule"))
    cmiss = cm.get("miss_diff", 0.0)
    for side, team, sign in (("h", home, 1), ("a", away, -1)):
        for name, val in miss_detail.get(side, []):
            m.append(_piece(f"{team} without {name}", sign * cmiss * val, "missing"))
    ci = cm.get("imp_diff", 0.0)
    if ci and abs(float(getattr(row, "imp_diff", 0.0))) > 0:
        ih, ia = getattr(row, "imp_h", None), getattr(row, "imp_a", None)
        lab = "Lineup strength from player impact ratings" + (f" ({home} {float(ih):+.1f}, {away} {float(ia):+.1f} per 48)" if ih is not None and ia is not None else "")
        m.append(_piece(lab, ci * float(row.imp_diff), "players"))
    cl = cm.get("lineup_diff", 0.0)
    if cl and abs(float(getattr(row, "lineup_diff", 0.0))) > 0:
        lh, la = getattr(row, "lineup_h", None), getattr(row, "lineup_a", None)
        lab = "Expected lineup, from players' box scores" + (f" ({home} {float(lh):.1f}, {away} {float(la):.1f}, Game Score per 40 across five players)" if lh is not None and la is not None else "")
        m.append(_piece(lab, cl * float(row.lineup_diff), "players"))
    t = [_piece("League baseline", ct.get("const", 0.0), "base"),
         _piece(f"Ratings: {pace:.1f} possessions at each side's efficiency", ct.get("r_total", 0.0) * float(row.r_total), "team")]
    if has_mk and "mk_total" in ct:
        t.append(_piece("Market's view of the teams (earlier closing totals)", ct["mk_total"] * float(row.mk_total), "market"))
    for k, lab in (("b2b_h", f"{home} on a back-to-back"), ("b2b_a", f"{away} on a back-to-back"), ("both_b2b", "Both on a back-to-back"),
                   ("miss_sum", "Players out, both teams")):
        v = ct.get(k, 0.0) * float(getattr(row, k, 0.0))
        if abs(v) >= 0.005:
            t.append(_piece(lab, v, "schedule" if k != "miss_sum" else "missing"))
    return {"margin": m, "total": t}


def team_stats(tb: pd.DataFrame, abbr_of: dict, poss_fn, only=None) -> dict:
    """Season-to-date four factors for and against, with league ranks (1 = best)."""
    tb = tb[tb.season_type == 2].copy()
    num = ["field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
           "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "total_turnovers", "turnovers", "team_score"]
    for c in num:
        tb[c] = pd.to_numeric(tb[c], errors="coerce").fillna(0)
    tb["tov"] = tb.total_turnovers.where(tb.total_turnovers > 0, tb.turnovers)
    opp = tb[["game_id", "team_id"] + num + ["tov"]].rename(columns={c: "o_" + c for c in num + ["tov", "team_id"]})
    x = tb.merge(opp, on="game_id")
    x = x[x.team_id != x.o_team_id]
    x = x[x.team_id.astype(int).isin(abbr_of)]
    s = x.groupby("team_id").sum(numeric_only=True)
    n = x.groupby("team_id").size()

    def side(p):      # p = "" for the team's offense, "o_" for what it allowed
        q = "o_" if p == "" else ""
        poss = poss_fn(s[p + "field_goals_attempted"], s[p + "offensive_rebounds"], s[p + "tov"], s[p + "free_throws_attempted"])
        return pd.DataFrame({
            "efg": 100 * (s[p + "field_goals_made"] + 0.5 * s[p + "three_point_field_goals_made"]) / s[p + "field_goals_attempted"],
            "tov": 100 * s[p + "tov"] / poss,
            "orb": 100 * s[p + "offensive_rebounds"] / (s[p + "offensive_rebounds"] + s[q + "defensive_rebounds"]),
            "ftr": 100 * s[p + "free_throws_made"] / s[p + "field_goals_attempted"],
            "tpr": 100 * s[p + "three_point_field_goals_attempted"] / s[p + "field_goals_attempted"],
            "ortg": 100 * s[p + "team_score"] / poss,
            "pace": poss / n,
        })
    off, dfn = side(""), side("o_")
    out = {}
    for tid in off.index:
        ab = abbr_of.get(int(tid))
        if not ab or (only is not None and ab not in only):
            continue
        ranks = {}
        for k, (_, better) in STATS.items():
            if better is None:
                ranks["off_" + k] = int(off[k].rank(ascending=False, method="min")[tid])
                ranks["def_" + k] = int(dfn[k].rank(ascending=False, method="min")[tid])
            else:
                ranks["off_" + k] = int(off[k].rank(ascending=not better, method="min")[tid])
                ranks["def_" + k] = int(dfn[k].rank(ascending=better, method="min")[tid])   # defense: allowing less is better
        out[ab] = {"games": int(n[tid]), "off": {k: round(float(off.at[tid, k]), 1) for k in STATS},
                   "def": {k: round(float(dfn.at[tid, k]), 1) for k in STATS}, "ranks": ranks}
    return out


def last_ten(games: pd.DataFrame, spreads: pd.DataFrame | None = None) -> dict:
    """Each team's last ten games: result straight up and against ESPN's spread where there is one."""
    g = games.copy()
    if spreads is not None and len(spreads):
        g = g.merge(spreads, on="game_id", how="left")
    else:
        g["espn_spread"] = np.nan
    rows = []
    for x in g.itertuples():
        for side in ("h", "a"):
            me, opp = (x.home, x.away) if side == "h" else (x.away, x.home)
            pf, pa = (x.pts_h, x.pts_a) if side == "h" else (x.pts_a, x.pts_h)
            sp = x.espn_spread if side == "h" else -x.espn_spread
            ats = None if pd.isna(sp) else ("W" if pf - pa > sp else "L" if pf - pa < sp else "P")
            site = "vs" if (side == "h" and not getattr(x, "neutral", False)) else ("at" if side == "a" and not getattr(x, "neutral", False) else "vs (N)")
            rows.append({"team": me, "date": x.date, "opp": opp, "site": site, "pf": int(pf), "pa": int(pa),
                         "line": None if pd.isna(sp) else float(-sp), "ats": ats})
    d = pd.DataFrame(rows).sort_values("date")
    out = {}
    for team, t in d.groupby("team"):
        t = t.tail(10).iloc[::-1]
        out[team] = [{"date": r.date.strftime("%Y-%m-%d"), "opp": r.opp, "site": r.site, "score": f"{r.pf}-{r.pa}",
                      "result": "W" if r.pf > r.pa else "L", "line": None if pd.isna(r.line) else float(r.line),
                      "ats": None if pd.isna(r.ats) else r.ats} for r in t.itertuples()]
    return out
