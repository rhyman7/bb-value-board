"""Situational spots for college games: letdown after an emotional win, lookahead to a key game,
back-to-back road games (and time-zone travel), and home underdogs in rivalry games.

Everything here uses only what is known before tip-off: results of earlier games, the schedule
(next games are on the schedule weeks ahead), ratings as of the game, and the line.

    spots(g, rated)   ->  one row per game with the home and away flags (spot_h / spot_a columns)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# standard-time UTC offsets, enough to count time zones crossed
TZ = {**{s: -5 for s in "CT DE DC FL GA IN ME MD MA MI NH NJ NY NC OH PA RI SC VT VA WV KY".split()},
      **{s: -6 for s in "AL AR IL IA KS LA MN MS MO NE ND OK SD TN TX WI".split()},
      **{s: -7 for s in "AZ CO ID MT NM UT WY".split()},
      **{s: -8 for s in "CA NV OR WA".split()},
      "AK": -9, "HI": -10, "PR": -4}
# full names, in case a season's file spells the state out
_NAMES = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
          "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
          "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
          "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
          "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
          "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
          "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
          "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
          "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
          "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR"}

UPSET = 5.0          # an "emotional" win: won as an underdog of this many points or more
FAV = 5.0            # lookahead only matters when the team is favored now by this much
TOP = 25             # a key game: next opponent rated in the top 25 at the time, or a rival
LOOK_DAYS = 5        # the key game is within this many days
RIVAL_SEASONS = 8    # rivals: same home state and met in at least this many seasons


def _state(x):
    if not isinstance(x, str):
        return None
    x = x.strip()
    return x if x in TZ else _NAMES.get(x)


def home_states(g):
    """Each team's home state: where most of its home games were played."""
    h = g[~g.neutral].assign(st=g.venue_address_state.map(_state)).dropna(subset=["st"])
    return h.groupby("home_id").st.agg(lambda s: s.value_counts().index[0]).to_dict()


def rivals(g, states):
    """Pairs of teams from the same state that met in at least RIVAL_SEASONS seasons."""
    p = g[["season", "home_id", "away_id"]].copy()
    p["a"] = p[["home_id", "away_id"]].min(axis=1)
    p["b"] = p[["home_id", "away_id"]].max(axis=1)
    n = p.drop_duplicates(["season", "a", "b"]).groupby(["a", "b"]).size()
    out = set()
    for (a, b), k in n.items():
        if k >= RIVAL_SEASONS and states.get(a) and states.get(a) == states.get(b):
            out.add((a, b))
    return out


def _team_view(g):
    """One row per team per game, from that team's side."""
    cols = ["game_id", "season", "date", "neutral", "venue_address_state", "close_spread", "r_margin", "margin", "net_h", "net_a"]
    h = g[cols + ["home_id", "away_id"]].rename(columns={"home_id": "tid", "away_id": "opp", "net_h": "net", "net_a": "opp_net"})
    h["site"] = np.where(h.neutral, "n", "h")
    a = g[cols + ["home_id", "away_id"]].rename(columns={"away_id": "tid", "home_id": "opp", "net_a": "net", "net_h": "opp_net"})
    a["site"] = np.where(a.neutral, "n", "a")
    for c in ["close_spread", "r_margin", "margin"]:
        a[c] = -a[c]
    t = pd.concat([h.assign(side="h"), a.assign(side="a")], ignore_index=True)
    t["exp"] = t.close_spread.fillna(t.r_margin)       # expected margin for this team: the line, else the ratings
    return t.sort_values(["season", "tid", "date", "game_id"]).reset_index(drop=True)


def spots(g: pd.DataFrame, rated: pd.DataFrame) -> pd.DataFrame:
    """g: games (data.all_games); rated: the same games with pre-game ratings (off/def per side) and
    r_margin. Returns game_id plus home/away spot flags and the rivalry flags."""
    df = rated.copy()
    df["net_h"] = df.off_h + df.def_h
    df["net_a"] = df.off_a + df.def_a
    states = home_states(g)
    rv = rivals(g, states)
    t = _team_view(df)
    t["st"] = t.venue_address_state.map(_state)
    t["home_tz"] = t.tid.map(states).map(TZ)
    t["tz"] = t.st.map(TZ)
    grp = t.groupby(["season", "tid"])

    # letdown: the team's last game (within 7 days) was a win as an underdog of UPSET or more, or a win over a rival
    t["prev_date"] = grp.date.shift(1)
    t["prev_win"] = grp.margin.shift(1) > 0
    t["prev_exp"] = grp.exp.shift(1)
    t["prev_opp"] = grp.opp.shift(1)
    t["prev_site"] = grp.site.shift(1)
    t["prev_tz"] = grp.tz.shift(1)
    recent = (t.date - t.prev_date).dt.days <= 7
    pair = lambda a, b: (min(a, b), max(a, b)) in rv
    t["rival"] = [pair(a, b) for a, b in zip(t.tid, t.opp)]
    prev_rival = [False if pd.isna(o) else pair(a, int(o)) for a, o in zip(t.tid, t.prev_opp)]
    t["letdown"] = recent & t.prev_win & ((t.prev_exp <= -UPSET) | pd.Series(prev_rival, index=t.index))

    # lookahead: favored by FAV+ now, and the next game (within LOOK_DAYS) is against a top-25 team or a rival
    t["next_date"] = grp.date.shift(-1)
    t["next_opp"] = grp.opp.shift(-1)
    # every team's net rating as of each date, ranked nationally, from its latest game on or before that date
    snap = t[["date", "tid", "net"]].sort_values("date")
    rank_on = {}
    latest = {}
    for d, day in snap.groupby("date"):
        for tid, net in zip(day.tid, day.net):
            latest[tid] = net
        vals = np.array(sorted(latest.values(), reverse=True))
        rank_on[d] = vals[TOP - 1] if len(vals) >= TOP else np.inf
    cut = t.date.map(rank_on)
    # the next opponent's rating as of today: its net in its own latest game on or before today
    nxt = t[["date", "next_opp"]].dropna().rename(columns={"next_opp": "tid"}).astype({"tid": int}).reset_index()
    own = t[["date", "tid", "net"]].sort_values("date")
    nxt = pd.merge_asof(nxt.sort_values("date"), own, on="date", by="tid", direction="backward").set_index("index")
    t["next_opp_net"] = nxt.net
    next_rival = [False if pd.isna(o) else pair(a, int(o)) for a, o in zip(t.tid, t.next_opp)]
    key = (t.next_opp_net >= cut) | pd.Series(next_rival, index=t.index)
    soon = (t.next_date - t.date).dt.days <= LOOK_DAYS
    t["lookahead"] = soon & key & (t.exp >= FAV) & ~t.rival

    # road trips: second straight true road game within 4 days; and crossing time zones
    road2 = (t.site == "a") & (t.prev_site == "a") & ((t.date - t.prev_date).dt.days <= 4)
    t["road_b2b"] = road2
    t["road_b2b_tz"] = road2 & (t.tz.notna() & t.prev_tz.notna()) & (t.tz != t.prev_tz)
    t["far_road"] = (t.site == "a") & ((t.tz - t.home_tz).abs() >= 2)

    flags = ["letdown", "lookahead", "road_b2b", "road_b2b_tz", "far_road"]
    w = t.pivot_table(index="game_id", columns="side", values=flags, aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.astype(float).fillna(0.0).reset_index()
    rv_game = t[t.side == "h"][["game_id", "rival"]]
    out = df[["game_id", "close_spread", "neutral"]].merge(w, on="game_id", how="left").merge(rv_game, on="game_id", how="left")
    out["rival"] = out.rival.fillna(False).astype(bool)
    out["rival_home_dog"] = (out.rival & ~out.neutral & (out.close_spread < 0)).astype(float)
    for f in flags:
        out[f"{f}_diff"] = out[f"{f}_h"] - out[f"{f}_a"]
    return out.drop(columns=["close_spread", "neutral"])


TESTS = {   # flag column -> (label, which side the spot says to take)
    "letdown_diff": ("Letdown after an emotional win", "fade"),
    "lookahead_diff": ("Lookahead to a key game", "fade"),
    "road_b2b_diff": ("Second straight road game in 4 days", "fade"),
    "road_b2b_tz_diff": ("Second straight road game, new time zone", "fade"),
    "far_road_diff": ("Road game 2+ time zones from home", "fade"),
    "rival_home_dog": ("Home underdog in a rivalry game", "back"),
}


def test(x: pd.DataFrame, col: str) -> dict:
    """Does the spot move the result against ESPN's line? Home-court games with a line only.
    pts: points the spot was worth against the spread in the direction the idea says (positive means
    it worked: the team in a letdown or lookahead spot fell short of the line, the home dog beat it),
    with its standard error;
    ats: the record taking the side the spot points to, in spot games."""
    x = x.dropna(subset=["close_spread"])
    x = x[~x.neutral]
    f = x[col].to_numpy(float)
    r = (x.margin - x.close_spread).to_numpy(float)
    n = int((f != 0).sum())
    if n < 30:
        return {"games": n}
    X = np.column_stack([np.ones_like(f), f])
    b, *_ = np.linalg.lstsq(X, r, rcond=None)
    cov = np.linalg.inv(X.T @ X) * (r - X @ b).var()
    sel = f != 0
    side = np.sign(f[sel]) * (-1 if TESTS[col][1] == "fade" else 1)
    res = np.sign(r[sel]) * side
    w, l, p = int((res > 0).sum()), int((res < 0).sum()), int((res == 0).sum())
    sign = -1 if TESTS[col][1] == "fade" else 1
    return {"games": n, "pts": round(sign * float(b[1]), 2), "se": round(float(np.sqrt(cov[1, 1])), 2),
            "ats": {"w": w, "l": l, "p": p, "pct": round(100 * w / max(1, w + l), 1), "se": round(100 * np.sqrt(0.25 / max(1, w + l)), 1)}}
