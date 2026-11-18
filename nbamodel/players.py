"""NBA player projections: minutes, points, rebounds, assists and threes made for every regular in a
game, built from minutes, per-minute production, the opponent and which teammates are out.

How a projection is built, using only games before the one projected:

  minutes   a recent average of the player's minutes when he plays (exponentially weighted), plus a
            share of the minutes his missing teammates usually play, capped at 42;
  rates     his points, rebounds, assists and threes per minute this season, with last season's
            numbers (half weight) and a league-average prior standing in early on;
  opponent  how many points, rebounds, assists and threes the opponent has allowed per game compared
            with the league, shrunk toward average early in the season;
  usage     when teammates are out, the production they usually carry is shared out to the players
            who are left (a fitted fraction of it, per stat).

"Missing" means the same as in availability.py: a regular in the team's last 15 games (played in
30%+ of them, 10+ minutes when he plays) who does not play. In the backtest the model knows who sat,
as the game model does; on the board it comes from the injury report.

    python -m nbamodel.players      # backtest: writes docs/nba/players_backtest.json
"""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from .data import CACHE, _get

STATS = ["points", "rebounds", "assists", "threes"]
WINDOW = 15
MAX_MIN = 42.0
PRIOR_MIN = 100.0       # minutes of league-average production a player's rates start from
PREV_W = 0.5            # weight on last season's minutes (capped at PREV_CAP)
PREV_CAP = 1000.0
OPP_GAMES = 10.0
FRESH = 5               # a missing player's minutes and production count by the share of the last 5 games he played        # league-average games an opponent's allowed numbers are shrunk with
ALPHAS = (0.15, 0.3)    # weights on the latest game in the two minutes averages
TUNE = [2021, 2022, 2023]
GRADE = [2024, 2025, 2026]
OUT = Path(__file__).resolve().parent.parent / "docs" / "nba"

# fitted on TUNE by tune() and saved in the backtest file; the board reads them from there
PARAMS_FILE = OUT / "players_backtest.json"
PARAMS = None


def box(season: int, refresh: bool = False) -> pd.DataFrame:
    p = pd.read_parquet(_get(f"player_box/parquet/player_box_{season}.parquet", CACHE / f"player_box_{season}.parquet", refresh))
    p = p[p.season_type == 2].copy()
    for c in ["minutes", "points", "rebounds", "assists", "three_point_field_goals_made"]:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
    p = p.rename(columns={"three_point_field_goals_made": "threes"})
    p = p[p.minutes > 0].dropna(subset=["athlete_id"])
    p["athlete_id"] = p.athlete_id.astype(int)
    p["team_id"] = p.team_id.astype(int)
    p["opponent_team_id"] = p.opponent_team_id.astype(int)
    p["game_id"] = p.game_id.astype(str)
    p["date"] = pd.to_datetime(p.game_date)
    keep = ["game_id", "date", "team_id", "opponent_team_id", "home_away", "athlete_id", "athlete_display_name",
            "team_abbreviation", "minutes"] + STATS
    return p[keep].sort_values(["date", "game_id"]).reset_index(drop=True)


class State:
    """Running numbers for one season: players' minutes and production, teams' recent rosters, and
    what each team has allowed."""

    def __init__(self, prev=None):
        self.ewm = {a: {} for a in ALPHAS}                 # alpha -> athlete -> minutes average
        self.tot = defaultdict(lambda: defaultdict(float))  # athlete -> stat -> sum (and "minutes", "games")
        self.prev = prev or {}                              # athlete -> stat -> per-minute rate, plus "minutes"
        self.last_team = {}
        self.names = {}
        self.recent = defaultdict(lambda: deque(maxlen=WINDOW))   # team -> deque of {athlete: minutes}
        self.allowed = defaultdict(lambda: defaultdict(float))    # team -> stat -> allowed sum, "games"
        self.league = defaultdict(float)                          # stat -> sum over team-games, "games"

    def rate(self, aid, stat):
        t, p = self.tot[aid], self.prev.get(aid)
        lg = self.league_rate(stat)
        pm = min(PREV_CAP, p["minutes"]) * PREV_W if p else 0.0
        pr = p[stat] if p else lg
        return (t[stat] + pr * pm + lg * PRIOR_MIN) / (t["minutes"] + pm + PRIOR_MIN)

    def league_rate(self, stat):
        # per player-minute: team totals over 240 minutes a game
        g = self.league["games"]
        return (self.league[stat] / (240.0 * g)) if g else {"points": 0.47, "rebounds": 0.18, "assists": 0.105, "threes": 0.052}[stat]

    def opp_factor(self, tid, stat):
        g = self.league["games"]
        if not g:
            return 1.0
        lg = self.league[stat] / g
        a = self.allowed[tid]
        return (a[stat] + lg * OPP_GAMES) / ((a["games"] + OPP_GAMES) * lg)

    def pool(self, tid):
        """Regulars and others who played for this team in its last WINDOW games: athlete -> (minutes
        when he plays, share of games played)."""
        games = self.recent[tid]
        mins = defaultdict(list)
        for gm in games:
            for aid, m in gm.items():
                mins[aid].append(m)
        n = max(len(games), 1)
        return {a: (float(np.mean(m)), len(m) / n) for a, m in mins.items() if self.last_team.get(a) == tid}

    def fresh(self, tid, n=FRESH):
        """How much of each player's absence is new: the share of the team's last n games he played.
        A player who has sat for a while is already out of his teammates' recent minutes."""
        games = list(self.recent[tid])[-n:]
        k = max(len(games), 1)
        c = defaultdict(int)
        for gm in games:
            for a in gm:
                c[a] += 1
        return {a: v / k for a, v in c.items()}

    def add_game(self, gp):
        """gp: the box score rows of one game (both teams)."""
        for tid, tp in gp.groupby("team_id"):
            self.recent[tid].append(dict(zip(tp.athlete_id, tp.minutes)))
            opp = int(tp.opponent_team_id.iloc[0])
            a = self.allowed[opp]
            a["games"] += 1
            self.league["games"] += 1
            for s in STATS:
                v = float(tp[s].sum())
                a[s] += v
                self.league[s] += v
            for r in tp.itertuples():
                t = self.tot[r.athlete_id]
                t["minutes"] += r.minutes
                t["games"] += 1
                for s in STATS:
                    t[s] += getattr(r, s)
                for al in ALPHAS:
                    e = self.ewm[al]
                    e[r.athlete_id] = r.minutes if r.athlete_id not in e else (1 - al) * e[r.athlete_id] + al * r.minutes
                self.last_team[r.athlete_id] = tid
                self.names[r.athlete_id] = r.athlete_display_name

    def carry(self):
        """This season's per-minute rates, for next season's prior."""
        out = {}
        for aid, t in self.tot.items():
            if t["minutes"] > 0:
                out[aid] = {"minutes": t["minutes"], **{s: t[s] / t["minutes"] for s in STATS}}
        return out


def regular(role, share):
    return share >= 0.3 and role >= 10


def team_rows(st: State, tid, opp, out_ids):
    """Projection inputs for one team in one game. out_ids: athletes known to be out (the regulars
    who sat, in the backtest; the injury report, live). Returns a list of dicts, one per available
    pool player."""
    pool = st.pool(tid)
    if not pool:
        return []
    rates = {a: {s: st.rate(a, s) for s in STATS} for a in pool}
    ew = {al: {a: st.ewm[al].get(a, pool[a][0]) for a in pool} for al in ALPHAS}
    base = ew[ALPHAS[-1]]
    miss = [a for a in pool if a in out_ids and regular(*pool[a])]
    fr = st.fresh(tid)
    w = {a: fr.get(a, 0.0) for a in miss}
    avail = [a for a in pool if a not in out_ids]
    if not avail:
        return []
    missing_min = sum(base[a] * w[a] for a in miss)
    avail_min = sum(base[a] for a in avail) or 1.0
    regs = [a for a in pool if regular(*pool[a])]
    share = {}
    for s in STATS:
        tot = sum(rates[a][s] * base[a] for a in regs) or 1.0
        share[s] = sum(rates[a][s] * base[a] * w[a] for a in miss) / tot
    last = st.recent[tid][-1] if st.recent[tid] else {}
    rows = []
    for a in avail:
        t = st.tot[a]
        rows.append({"athlete_id": a, "team_id": tid, "opp_id": opp, "name": st.names.get(a, str(a)),
                     **{f"ewm{int(al * 100)}": ew[al][a] for al in ALPHAS},
                     "role": pool[a][0], "share": pool[a][1], "regular": regular(*pool[a]),
                     "last_min": float(last.get(a, 0.0)), "miss_ratio": missing_min / avail_min,
                     **{f"rate_{s}": rates[a][s] for s in STATS},
                     **{f"opp_{s}": st.opp_factor(opp, s) for s in STATS},
                     **{f"mshare_{s}": share[s] for s in STATS},
                     "base_games": t["games"],
                     **{f"avg_{s}": (t[s] / t["games"]) if t["games"] else np.nan for s in STATS + ["minutes"]},
                     "out": [st.names.get(m, str(m)) for m in miss]})
    return rows


def load_params():
    return json.loads(PARAMS_FILE.read_text())["params"]


def min_features(df):
    e = df[f"ewm{int(ALPHAS[-1] * 100)}"]
    return pd.DataFrame({"const": 1.0, **{f"ewm{int(al * 100)}": df[f"ewm{int(al * 100)}"] for al in ALPHAS},
                         "role": df.role, "season_avg": df.avg_minutes.fillna(df.role), "last_game": df.last_min,
                         "teammates_out": e * df.miss_ratio}, index=df.index)


def fit_minutes(df):
    """Least squares for minutes on the tuning seasons, players who played."""
    d = df[df.a_minutes.notna()]
    X = min_features(d).to_numpy(float)
    b = np.linalg.lstsq(X, d.a_minutes.to_numpy(float), rcond=None)[0]
    return dict(zip(min_features(d.head(1)).columns, (float(v) for v in b)))


def project(df: pd.DataFrame, params=None) -> pd.DataFrame:
    """Turn team_rows() inputs into numbers."""
    params = params or load_params()
    df = df.copy()
    X = min_features(df)
    df["p_minutes"] = np.clip(sum(X[k] * v for k, v in params["minutes"].items()), 0.0, MAX_MIN)
    for s in STATS:
        df[f"p_{s}"] = (df[f"rate_{s}"] * df.p_minutes * df[f"opp_{s}"] ** params["beta"][s]
                        * (1 + params["u"][s] * df[f"mshare_{s}"]))
    # last step: a straight-line correction fitted on the tuning seasons, leaning a little on the
    # season average (the rates above are shrunk toward the league, which pulls stars down)
    for s, (c0, c1, c2) in params.get("cal", {}).items():
        df[f"p_{s}"] = np.maximum(0.0, c0 + c1 * df[f"p_{s}"] + c2 * df[f"avg_{s}"].fillna(df[f"p_{s}"]))
    df["p_minutes"] = df.p_minutes.clip(upper=MAX_MIN)
    return df


def walk(seasons, first_prev=None):
    """Walk forward through the seasons: projection inputs for every pool player of every game, with
    what he actually did (NaN when he did not play). Players with no game this season have no row."""
    rows = []
    prev = first_prev
    for season in seasons:
        try:
            b = box(season)
        except Exception:
            continue
        st = State(prev)
        for day, todays in b.groupby("date"):
            for gid, gp in todays.groupby("game_id"):
                played = set(gp.athlete_id)
                actual = gp.set_index("athlete_id")
                for tid, tp in gp.groupby("team_id"):
                    opp = int(tp.opponent_team_id.iloc[0])
                    out_ids = {a for a in st.pool(tid) if a not in played}
                    for r in team_rows(st, tid, opp, out_ids):
                        r.update({"season": season, "game_id": gid, "date": day})
                        a = r["athlete_id"]
                        for s in STATS + ["minutes"]:
                            r[f"a_{s}"] = float(actual.at[a, s]) if a in played else np.nan
                        rows.append(r)
            for gid, gp in todays.groupby("game_id"):
                st.add_game(gp)
        prev = st.carry()
    return pd.DataFrame(rows)


def mae(x, s, col):
    return float((x[f"a_{s}"] - x[col]).abs().mean())


def grade(x, params):
    """Players who played, projected for 15+ minutes. Model vs his season average so far."""
    p = graded(project(x, params))
    out = {"player_games": int(len(p))}
    for s in ["minutes"] + STATS:
        d = (p[f"a_{s}"] - p[f"p_{s}"]).abs() - (p[f"a_{s}"] - p[f"avg_{s}"]).abs()
        out[s] = {"model_mae": round(mae(p, s, f"p_{s}"), 3), "season_avg_mae": round(mae(p, s, f"avg_{s}"), 3),
                  "gain": round(float(-d.mean()), 3), "gain_se": round(float(d.std() / np.sqrt(len(d))), 3),
                  "model_bias": round(float((p[f"p_{s}"] - p[f"a_{s}"]).mean()), 3)}
    return out


def graded(p):
    return p[p.a_minutes.notna() & (p.role >= 15) & (p.base_games >= 3)]


def tune(x):
    """Fit the minutes model, then the opponent and usage weights per stat, on the tuning seasons."""
    t = x[x.season.isin(TUNE)]
    params = {"minutes": fit_minutes(t), "beta": {s: 0.0 for s in STATS}, "u": {s: 0.0 for s in STATS}}
    p = graded(project(t, params))
    for s in STATS:
        best = None
        for beta in (0.0, 0.5, 1.0, 1.5):
            for uu in (0.0, 0.25, 0.5, 0.75, 1.0):
                pred = p[f"p_{s}"] * p[f"opp_{s}"] ** beta * (1 + uu * p[f"mshare_{s}"])
                e = float((p[f"a_{s}"] - pred).abs().mean())
                if best is None or e < best[0]:
                    best = (e, beta, uu)
        params["beta"][s], params["u"][s] = best[1], best[2]
    p = graded(project(t, params))
    params["cal"] = {}
    for s in ["minutes"] + STATS:
        X = np.column_stack([np.ones(len(p)), p[f"p_{s}"], p[f"avg_{s}"]])
        params["cal"][s] = [round(float(v), 4) for v in np.linalg.lstsq(X, p[f"a_{s}"].to_numpy(float), rcond=None)[0]]
    params["minutes"] = {k: round(v, 4) for k, v in params["minutes"].items()}
    return params


def run():
    seasons = list(range(min(TUNE) - 1, max(GRADE) + 1))
    x = walk(seasons)
    params = tune(x)
    summary = {
        "params": params,
        "tune_seasons": f"{TUNE[0]}-{TUNE[-1]}", "holdout_seasons": f"{GRADE[0]}-{GRADE[-1]}",
        "tune": grade(x[x.season.isin(TUNE)], params),
        "holdout": grade(x[x.season.isin(GRADE)], params),
        "by_season": {int(s): grade(g, params) for s, g in x[x.season.isin(GRADE)].groupby("season")},
        "teammates_out": grade(x[x.season.isin(GRADE) & (x.miss_ratio > 0.1)], params),
    }
    print(json.dumps({k: summary[k] for k in ["params", "tune", "holdout", "teammates_out"]}, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "players_backtest.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---- tonight's projections, the nightly log and its grades -------------------------------------

LOG = Path(__file__).resolve().parent.parent / "docs" / "players" / "nba_log.csv"
GRADES = LOG.parent / "nba_grades.json"
LOG_COLS = ["date", "game_id", "athlete_id", "name", "team", "opp", "out_on_team"] + \
           [f"p_{s}" for s in ["minutes"] + STATS] + [f"avg_{s}" for s in ["minutes"] + STATS]


def state_today(season, refresh=True):
    """Running numbers as of today: last season for the prior, then every game this season."""
    prev = None
    try:
        st = State()
        for _, gp in box(season - 1).groupby("game_id", sort=False):
            st.add_game(gp)
        prev = st.carry()
    except Exception:
        pass
    st = State(prev)
    for _, gp in box(season, refresh=refresh).groupby(["date", "game_id"]):
        st.add_game(gp)
    return st


def tonight(st, games, out_ids, abbr):
    """games: [(game_id, home_id, away_id)]; out_ids: athletes listed Out (or Doubtful). Returns
    {game_id: {"home": [...], "away": [...], "out": {"home": [...], "away": [...]}}}."""
    params = load_params()
    res = {}
    for gid, hid, aid in games:
        entry = {"out": {}}
        for side, tid, opp in (("home", hid, aid), ("away", aid, hid)):
            rows = team_rows(st, tid, opp, out_ids)
            if not rows:
                entry[side] = []
                entry["out"][side] = []
                continue
            p = project(pd.DataFrame(rows), params).sort_values("p_minutes", ascending=False)
            p = p[p.p_minutes >= 8]
            entry["out"][side] = rows[0]["out"]
            entry[side] = [{"id": int(r.athlete_id), "name": r.name, "team": abbr.get(tid, ""), "opp": abbr.get(opp, ""),
                            **{k: round(float(getattr(r, f"p_{k}")), 1) for k in ["minutes"] + STATS},
                            "avg": {k: (None if pd.isna(getattr(r, f"avg_{k}")) else round(float(getattr(r, f"avg_{k}")), 1))
                                    for k in ["minutes"] + STATS}}
                           for r in p.itertuples()]
        res[gid] = entry
    return res


def log(day, proj):
    """Save tonight's projections (today's games only). A later run the same day replaces them, so
    the log keeps the last projection before tip-off."""
    rows = []
    for gid, e in proj.items():
        for side in ("home", "away"):
            for r in e.get(side, []):
                rows.append({"date": day.isoformat(), "game_id": str(gid), "athlete_id": r["id"], "name": r["name"],
                             "team": r["team"], "opp": r["opp"], "out_on_team": "; ".join(e["out"].get(side, [])),
                             **{f"p_{k}": r[k] for k in ["minutes"] + STATS},
                             **{f"avg_{k}": r["avg"][k] for k in ["minutes"] + STATS}})
    new = pd.DataFrame(rows, columns=LOG_COLS)
    if LOG.exists():
        old = pd.read_csv(LOG, dtype={"game_id": str})
        old = old[~old.game_id.isin(set(new.game_id))]
        new = pd.concat([old, new], ignore_index=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["date", "game_id", "team", "p_minutes"], ascending=[True, True, True, False]).to_csv(LOG, index=False)
    return len(rows)


def grade_log(season):
    """Grade every logged projection whose game is in the box scores: the model against each player's
    season average going in. Players who did not play are left out (a prop on them would be void)."""
    if not LOG.exists():
        return None
    lg = pd.read_csv(LOG, dtype={"game_id": str})
    b = box(season, refresh=False)[["game_id", "athlete_id", "minutes"] + STATS]
    m = lg.merge(b.rename(columns={k: f"a_{k}" for k in ["minutes"] + STATS}), on=["game_id", "athlete_id"], how="inner")
    if m.empty:
        return {"nights": 0}
    m["date"] = pd.to_datetime(m.date)
    last = m.date.max()

    def summ(x):
        out = {"player_games": int(len(x))}
        for k in ["minutes"] + STATS:
            y = x.dropna(subset=[f"avg_{k}"])
            out[k] = {"model_mae": round(float((y[f"a_{k}"] - y[f"p_{k}"]).abs().mean()), 2),
                      "season_avg_mae": round(float((y[f"a_{k}"] - y[f"avg_{k}"]).abs().mean()), 2),
                      "model_closer_pct": round(100 * float(((y[f"a_{k}"] - y[f"p_{k}"]).abs()
                                                             < (y[f"a_{k}"] - y[f"avg_{k}"]).abs()).mean()), 1)}
        return out

    night = m[m.date == last]
    top = night.assign(err=(night.a_points - night.p_points).abs()).sort_values("p_points", ascending=False).head(12)
    res = {
        "nights": int(m.date.nunique()), "lastNight": last.date().isoformat(),
        "last_night": summ(night), "last_7_days": summ(m[m.date > last - pd.Timedelta(days=7)]), "season": summ(m),
        "last_night_top": [{"name": r.name, "team": r.team, "opp": r.opp, "p_points": r.p_points, "a_points": r.a_points,
                            "p_rebounds": r.p_rebounds, "a_rebounds": r.a_rebounds, "p_assists": r.p_assists, "a_assists": r.a_assists,
                            "p_threes": r.p_threes, "a_threes": r.a_threes} for r in top.itertuples()],
    }
    GRADES.write_text(json.dumps(res, indent=1, allow_nan=False))
    return res


if __name__ == "__main__":
    run()
