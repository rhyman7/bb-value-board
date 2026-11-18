"""Nightly matchup report PDFs for the NBA and college slates, in the style of the football
weekly matchup reports.

Every number comes from two files the boards already publish:
  docs/feed/<league>.json (+ cbb_players.json)   game projections and player projections (this repo)
  <dashboard>/data/data.json                      team colors, records, team stats, defense vs position

    python -m reports.matchup_pdf --league nba [--date 2026-11-18] [--dash path-or-url]

writes docs/reports/<league>/<date>.pdf, docs/reports/<league>/latest.pdf and docs/reports/index.json.
These are projections, not picks: the models do not beat the closing line.
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (CondPageBreak, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "reports"
ET = timezone(timedelta(hours=-5))   # tip times are shown in Eastern; the feed's tips carry their own offset

LEAGUES = {
    "nba": {
        "name": "NBA", "feed": "docs/feed/nba.json", "players": None,
        "dash": "https://raw.githubusercontent.com/rhyman7/nba/main/data/data.json",
        "board": "https://rhyman7.github.io/bb-value-board/nba/", "min_minutes": 20.0, "min_points": 10.0,
        "spots_per_game": 3, "player_table": 5, "compact": False,
    },
    "cbb": {
        "name": "College Basketball", "feed": "docs/feed/cbb.json", "players": "docs/feed/cbb_players.json",
        "dash": "https://raw.githubusercontent.com/rhyman7/cbb/main/data/data.json",
        "board": "https://rhyman7.github.io/bb-value-board/cbb/", "min_minutes": 20.0, "min_points": 8.0,
        "spots_per_game": 2, "player_table": 0, "compact": True,
    },
}
POS_NAME = {"G": "guards", "F": "forwards", "C": "centers"}

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#5b6470")
RULE = colors.HexColor("#d9dee4")
GOOD = colors.HexColor("#1f7a3e")
BAD = colors.HexColor("#b3261e")
PANEL = colors.HexColor("#f4f6f8")

S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Helvetica", fontSize=11, leading=15, textColor=MUTED),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK, spaceBefore=6, spaceAfter=6),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=INK, spaceBefore=4, spaceAfter=2),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12, textColor=INK),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8, leading=10.5, textColor=MUTED),
    "note": ParagraphStyle("note", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8, leading=10, textColor=INK),
    "cellc": ParagraphStyle("cellc", fontName="Helvetica", fontSize=8, leading=10, textColor=INK, alignment=TA_CENTER),
    "spot": ParagraphStyle("spot", fontName="Helvetica", fontSize=8.5, leading=11, textColor=INK, leftIndent=8),
}


# ---------------------------------------------------------------- data

def load(src: str):
    if src.startswith("http"):
        req = urllib.request.Request(src, headers={"User-Agent": "bb-value-board-report"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    return json.loads(Path(src).read_text())


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def most_rank(rank: int, n: int) -> str:
    """Rank 1 = allows the fewest. Turn it into 'the 3rd-most' / 'the fewest' wording."""
    if rank <= n / 2:
        return "the fewest" if rank == 1 else f"the {ordinal(rank)}-fewest"
    k = n - rank + 1
    return "the most" if k == 1 else f"the {ordinal(k)}-most"


def team_ranks(teams: dict) -> dict:
    """League ranks for the team stats the report quotes (1 = best)."""
    keys = {  # (section, field, higher is better)
        "ortg": ("offense", "adjOrtg", True), "drtg": ("defense", "adjDrtg", False),
        "efg": ("offense", "efg", True), "oefg": ("defense", "efg", False),
        "tpct": ("offense", "threePct", True), "otpct": ("defense", "threePct", False),
        "tov": ("offense", "tovPct", False), "otov": ("defense", "tovPct", True),
        "orb": ("offense", "orbPct", True), "drb": ("defense", "drbPct", True),
        "pace": (None, "pace", True), "net": (None, "adjNet", True),
    }
    out = {a: {} for a in teams}
    for k, (sec, f, hi) in keys.items():
        vals = [(a, (t.get(sec) or {}).get(f) if sec else t.get(f)) for a, t in teams.items()]
        vals = [(a, v) for a, v in vals if v is not None]
        vals.sort(key=lambda x: -x[1] if hi else x[1])
        for i, (a, _) in enumerate(vals, 1):
            out[a][k] = i
    return out


def tip_et(g, dash_game):
    t = g.get("tip") or (dash_game or {}).get("start")
    if not t:
        return "TBD"
    dt = datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(ET)
    if dash_game is not None and dash_game.get("timeValid") is False:
        return "TBD"
    return dt.strftime("%-I:%M %p ET")


# ---------------------------------------------------------------- spots

def player_spots(g, plist, dash, min_minutes, min_points):
    """Score each rotation player's matchup from the opponent's defense against his position and his
    projection against his season average. Positive = good spot."""
    teams = dash["teams"]
    # rank defense vs position among the dashboard's own teams (1 = allows the fewest points)
    vpos = {}
    for pos in POS_NAME:
        rows = sorted((v["pts"], a) for a, t in teams.items() if (v := (t.get("vsPos") or {}).get(pos)))
        for i, (_, a) in enumerate(rows, 1):
            vpos[(a, pos)] = {**teams[a]["vsPos"][pos], "rank": i, "of": len(rows)}
    out = []
    for side in ("home", "away"):
        for p in plist.get(side, []):
            if (p.get("minutes") or 0) < min_minutes or (p.get("points") or 0) < min_points:
                continue
            team, opp = p["team"], p["opp"]
            dp = next((x for x in (teams.get(team) or {}).get("players", []) if str(x.get("id")) == str(p["id"])), None)
            pos = (dp or {}).get("pos")
            vp = vpos.get((opp, pos))
            avg = (p.get("avg") or {}).get("points")
            if avg is None or vp is None:
                continue
            n = vp["of"]
            pos_z = (vp["rank"] - (n + 1) / 2) / (n / 2)               # -1 stingiest .. +1 most generous
            delta = p["points"] - avg
            proj_z = max(-1.0, min(1.0, delta / max(avg, 6.0) / 0.25))  # +/-25% of his average = +/-1
            score = 0.6 * pos_z + 0.4 * proj_z
            out.append({**p, "pos": pos, "vp": vp, "score": score, "delta": delta, "avg_pts": avg,
                        "l5": (dp or {}).get("l5pts"), "n": n, "side": side})
    return out


def is_good(s):
    return s["score"] > 0.15 and s["vp"]["rank"] > s["n"] * 2 / 3


def is_bad(s):
    return s["score"] < -0.15 and s["vp"]["rank"] <= s["n"] / 3


def spot_reason(s, good: bool):
    vp, n, pos = s["vp"], s["n"], s["pos"]
    d = f"{s['opp']} allow {most_rank(vp['rank'], n)} points to {POS_NAME[pos]} ({vp['pts']:.1f}/g)"
    sign = "+" if s["delta"] >= 0 else "-"
    proj = f"proj. {s['points']:.1f} pts vs {s['avg_pts']:.1f} avg ({sign}{abs(s['delta']):.1f})"
    extra = ""
    if s.get("l5") is not None:
        l5 = s["l5"]
        if good and l5 >= s["avg_pts"] + 3:
            extra = f"; {l5:.1f} over his last 5"
        elif not good and l5 <= s["avg_pts"] - 3:
            extra = f"; {l5:.1f} over his last 5"
    return f"{d}; {proj}{extra}"


# ---------------------------------------------------------------- drawing

def hexcol(h, default="#444444"):
    h = (h or "").strip("#")
    try:
        return colors.HexColor("#" + h) if len(h) == 6 else colors.HexColor(default)
    except Exception:
        return colors.HexColor(default)


def readable_alt(t):
    """Secondary color for the lettering, falling back to white when it would vanish on the primary."""
    c, a = hexcol(t.get("color")), hexcol(t.get("alt"), "#ffffff")
    lum = lambda x: 0.299 * x.red + 0.587 * x.green + 0.114 * x.blue
    return a if abs(lum(c) - lum(a)) > 0.25 else (colors.white if lum(c) < 0.55 else INK)


def team_block(abbr, t, w=0.8 * inch, h=0.36 * inch, fs=13):
    st = ParagraphStyle("tb", fontName="Helvetica-Bold", fontSize=fs, leading=fs + 2, alignment=TA_CENTER,
                        textColor=readable_alt(t))
    tb = Table([[Paragraph(abbr, st)]], colWidths=[w], rowHeights=[h])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), hexcol(t.get("color"))),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROUNDEDCORNERS", [4, 4, 4, 4])]))
    return tb


def record_str(t, lg):
    r = t.get("record") or {}
    s = f"{r.get('w', 0)}-{r.get('l', 0)}"
    if lg == "cbb" and t.get("rank"):
        s = f"#{t['rank']} · " + s
    return s


def game_header(g, dash, dgame, lg, width):
    teams = dash["teams"]
    a, h = teams.get(g["away"], {}), teams.get(g["home"], {})
    name = ParagraphStyle("nm", parent=S["body"], fontName="Helvetica-Bold", fontSize=11, leading=13)
    meta = ParagraphStyle("mt", parent=S["small"], alignment=2)
    at = "vs" if g.get("neutral") or (dgame or {}).get("neutral") else "@"
    bits = [tip_et(g, dgame)]
    if dgame and dgame.get("tv"):
        bits.append(dgame["tv"])
    if dgame and dgame.get("note"):
        bits.append(dgame["note"])
    if dgame and dgame.get("arena") and (g.get("neutral") or dgame.get("neutral")):
        bits.append(dgame["arena"])
    row = [team_block(g["away"], a),
           Paragraph(f"{a.get('short') or g['away']}<br/><font size=8 color='#5b6470'>{record_str(a, lg)}</font>", name),
           Paragraph(f"<font size=14 color='#5b6470'>{at}</font>", ParagraphStyle("at", parent=S["body"], alignment=TA_CENTER)),
           team_block(g["home"], h),
           Paragraph(f"{h.get('short') or g['home']}<br/><font size=8 color='#5b6470'>{record_str(h, lg)}</font>", name),
           Paragraph(" · ".join(bits), meta)]
    cw = [0.85 * inch, 1.45 * inch, 0.4 * inch, 0.85 * inch, 1.45 * inch]
    cw.append(width - sum(cw))
    tb = Table([row], colWidths=cw)
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, -1), 1.2, INK),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 2)]))
    return tb


def fmt_spread(team, m):
    """m = projected home margin. Returned as the favourite's number."""
    if m is None:
        return "-"
    return f"{team} -{abs(m):.1f}" if abs(m) >= 0.05 else "Pick'em"


def projection_strip(g, width):
    m = g.get("projection", g.get("margin"))
    fav = g["home"] if (m or 0) >= 0 else g["away"]
    wp = g.get("win_prob")
    wp_fav = wp if (m or 0) >= 0 else (1 - wp if wp is not None else None)
    cells = [("Projected spread", fmt_spread(fav, m)),
             ("Projected total", f"{g.get('projection_total', g.get('total')):.1f}" if g.get("total") is not None else "-"),
             ("Win chance", f"{fav} {wp_fav * 100:.0f}%" if wp_fav is not None else "-"),
             ("Pace", f"{g['pace']:.1f} poss." if g.get("pace") else "-")]
    ms, mt = g.get("mkt_spread"), g.get("mkt_total")
    if ms is not None or mt is not None:
        mk = []
        if ms is not None:
            mk.append(f"{g['home']} {ms:+.1f}")
        if mt is not None:
            mk.append(f"o/u {mt:.1f}")
        cells.append(("Market (ESPN)", "  ".join(mk)))
    lab = ParagraphStyle("lb", parent=S["small"], fontSize=7, leading=9, alignment=TA_CENTER)
    val = ParagraphStyle("vl", parent=S["body"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, alignment=TA_CENTER)
    row = [[Paragraph(k.upper(), lab), Paragraph(v, val)] for k, v in cells]
    cw = width / len(row)
    tb = Table([[Table([[r[0]], [r[1]]], colWidths=[cw - 6]) for r in row]], colWidths=[cw] * len(row))
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), PANEL), ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                            ("INNERGRID", (0, 0), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return tb


def stat_table(g, dash, ranks, width, compact=False):
    a, h = g["away"], g["home"]
    ta, th = dash["teams"].get(a, {}), dash["teams"].get(h, {})
    n = len(dash["teams"])
    def v(t, sec, f, fmt="{:.1f}"):
        x = (t.get(sec) or {}).get(f) if sec else t.get(f)
        return fmt.format(x) if x is not None else "-"
    def rk(abbr, k):
        r = ranks.get(abbr, {}).get(k)
        return f" <font color='#5b6470' size=7>({ordinal(r)})</font>" if r else ""
    rows = [("Net rating", None, "adjNet", "net"), ("Offense rating", "offense", "adjOrtg", "ortg"),
            ("Defense rating", "defense", "adjDrtg", "drtg"), ("Effective FG %", "offense", "efg", "efg"),
            ("Opp. eFG % allowed", "defense", "efg", "oefg"), ("3-point %", "offense", "threePct", "tpct"),
            ("Opp. 3P % allowed", "defense", "threePct", "otpct"), ("Turnover %", "offense", "tovPct", "tov"),
            ("Off. rebound %", "offense", "orbPct", "orb"), ("Pace", None, "pace", "pace")]
    if compact:
        rows = [r for r in rows if r[3] in ("net", "ortg", "drtg", "tpct", "otpct", "pace")]
    data = [[Paragraph(f"<b>{a}</b>", S["cellc"]), Paragraph(f"<font size=7 color='#5b6470'>rank of {n}</font>", S["cellc"]),
             Paragraph(f"<b>{h}</b>", S["cellc"])]]
    for lab, sec, f, k in rows:
        data.append([Paragraph(v(ta, sec, f) + rk(a, k), S["cellc"]), Paragraph(lab, S["cellc"]),
                     Paragraph(v(th, sec, f) + rk(h, k), S["cellc"])])
    tb = Table(data, colWidths=[width * 0.29, width * 0.42, width * 0.29])
    tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    return tb


def out_line(g, plist):
    out = (plist or {}).get("out") or {}
    miss = g.get("missing") or {}
    parts = []
    for side in ("away", "home"):
        names = out.get(side) or (miss.get(f"{side}_names") if isinstance(miss, dict) else None) or \
            (miss.get(side) if isinstance(miss, dict) and isinstance(miss.get(side), list) else None) or []
        if names:
            parts.append(f"<b>{g[side]}:</b> " + ", ".join(names))
    return Paragraph("<b>Out:</b> " + ("; ".join(parts) if parts else "nobody listed"), S["small"])


def spots_block(spots, k, width):
    good = sorted([s for s in spots if is_good(s)], key=lambda s: -s["score"])[:k]
    bad = sorted([s for s in spots if is_bad(s)], key=lambda s: s["score"])[:k]
    def lines(lst, good_):
        if not lst:
            return [Paragraph("None stand out.", S["small"])]
        return [Paragraph(f"<b>{s['name']}</b> <font color='#5b6470'>({s['team']} {s['pos']})</font>: {spot_reason(s, good_)}",
                          S["spot"]) for s in lst]
    hg = ParagraphStyle("hg", parent=S["h3"], textColor=GOOD)
    hb = ParagraphStyle("hb", parent=S["h3"], textColor=BAD)
    return [Paragraph("Good spots", hg), *lines(good, True), Paragraph("Tough spots", hb), *lines(bad, False)], good


def player_table(g, plist, k, width):
    hdr = ["Player", "Min", "Pts", "Reb", "Ast", "3PM"]
    data = []
    for side in ("away", "home"):
        ps = sorted(plist.get(side, []), key=lambda p: -(p.get("minutes") or 0))[:k]
        for p in ps:
            def c(s):
                v, a = p.get(s), (p.get("avg") or {}).get(s)
                txt = f"{v:.1f}" if v is not None else "-"
                if a is not None:
                    txt += f" <font size=6.5 color='#5b6470'>{a:.1f}</font>"
                pr = (p.get("props") or {}).get(s)
                if pr and pr.get("line") is not None:
                    txt += f" <font size=6.5 color='#1d4ed8'>L{pr['line']:g}</font>"
                return Paragraph(txt, S["cellc"])
            data.append([Paragraph(f"<b>{p['team']}</b> {p['name']}", S["cell"]), c("minutes"), c("points"),
                         c("rebounds"), c("assists"), c("threes")])
    if not data:
        return None
    data.insert(0, [Paragraph(f"<b>{x}</b>", S["cellc"] if i else S["cell"]) for i, x in enumerate(hdr)])
    cw = [width * 0.34] + [width * 0.132] * 5
    tb = Table(data, colWidths=cw, repeatRows=1)
    tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2)]))
    return tb


def ranked_table(rows, cols, widths):
    data = [[Paragraph(f"<b>{c}</b>", S["cell"]) for c in cols]] + \
           [[Paragraph(str(x), S["cell"]) for x in r] for r in rows]
    tb = Table(data, colWidths=widths, repeatRows=1)
    tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return tb


# ---------------------------------------------------------------- report

def build(lg: str, day: str | None = None, dash_src: str | None = None, out_dir: Path = OUT) -> Path | None:
    cfg = LEAGUES[lg]
    feed = load(str(ROOT / cfg["feed"]))
    pfeed = load(str(ROOT / cfg["players"]))["games"] if cfg["players"] else None
    dash = load(dash_src or cfg["dash"])
    day = day or datetime.now(ET).date().isoformat()
    games = [g for g in feed["games"] if g["date"] == day]
    if not games:
        print(f"{lg}: no games on {day}")
        return None
    games.sort(key=lambda g: (g.get("tip") or "", g["home"]))
    dgames = {str(x["id"]): x for x in dash.get("games", [])}
    ranks = team_ranks(dash["teams"])

    out_dir = Path(out_dir) / lg
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{day}.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=0.55 * inch, rightMargin=0.55 * inch,
                            topMargin=0.55 * inch, bottomMargin=0.6 * inch,
                            title=f"{cfg['name']} Matchup Report, {day}", author="bb-value-board")
    W = doc.width
    nice_day = datetime.fromisoformat(day).strftime("%A, %B %-d, %Y")
    gen = datetime.fromisoformat(feed["generatedAt"]).astimezone(ET).strftime("%b %-d, %-I:%M %p ET")

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(MUTED)
        c.drawString(doc.leftMargin, 0.35 * inch, f"{cfg['name']} Matchup Report · {nice_day} · projections, not picks")
        c.drawRightString(doc.leftMargin + W, 0.35 * inch, f"Page {d.page}")
        c.restoreState()

    story = [Paragraph(f"{cfg['name']} Matchup Report", S["title"]),
             Paragraph(f"{nice_day} · {len(games)} games · data as of {gen}", S["sub"]), Spacer(1, 8),
             Paragraph("These are the model's projections, not picks. The model does not beat the closing line yet, "
                       "so use the numbers as a read on each matchup, not as bets. Good and tough spots rank rotation "
                       "players by how many points the opponent allows to their position and by their projection "
                       "against their season average.", S["note"]),
             Spacer(1, 10), Paragraph("Tonight's slate", S["h2"])]

    # slate table + collect spots
    all_spots, slate_rows, sections = [], [], []
    for g in games:
        dg = dgames.get(str(g["id"]))
        plist = (pfeed or {}).get(str(g["id"])) if pfeed is not None else g.get("players")
        plist = plist or {}
        spots = player_spots(g, plist, dash, cfg["min_minutes"], cfg["min_points"])
        all_spots += spots
        m = g.get("projection", g.get("margin"))
        fav = g["home"] if (m or 0) >= 0 else g["away"]
        wp = g.get("win_prob")
        wpf = wp if (m or 0) >= 0 else (1 - wp if wp is not None else None)
        at = "vs" if g.get("neutral") else "@"
        slate_rows.append([tip_et(g, dg), f"{g['away']} {at} {g['home']}", fmt_spread(fav, m),
                           f"{g.get('projection_total', g.get('total')):.1f}", f"{wpf * 100:.0f}%" if wpf is not None else "-",
                           (f"{g['home']} {g['mkt_spread']:+.1f}" if g.get("mkt_spread") is not None else "-")])
        sections.append((g, dg, plist, spots))
    story.append(ranked_table(slate_rows, ["Tip", "Game", "Proj. spread", "Proj. total", "Fav. win %", "Market spread"],
                              [W * 0.14, W * 0.24, W * 0.18, W * 0.14, W * 0.12, W * 0.18]))

    # summary lists (football: Best Overall Matchups, Most Likely to Score a TD)
    story += [PageBreak(), Paragraph("Best overall matchups", S["h2"])]
    # a good spot matters more for a bigger part of the offense
    best = sorted([s for s in all_spots if is_good(s)], key=lambda s: -s["score"] * (0.5 + 0.5 * min(1.0, s["points"] / 20)))[:15]
    story.append(ranked_table([[i, f"<b>{s['name']}</b> ({s['team']} {s['pos']})", f"vs {s['opp']}", spot_reason(s, True)]
                               for i, s in enumerate(best, 1)], ["#", "Player", "Opp", "Why"],
                              [W * 0.05, W * 0.27, W * 0.09, W * 0.59]))
    story += [Spacer(1, 10), Paragraph("Top projected scorers", S["h2"])]
    scorers = sorted(all_spots, key=lambda s: -s["points"])[:15]
    def line(s, k):
        pr = (s.get("props") or {}).get(k)
        return f"{pr['line']:g}" if pr and pr.get("line") is not None else "-"
    story.append(ranked_table([[i, f"<b>{s['name']}</b> ({s['team']})", f"vs {s['opp']}", f"{s['points']:.1f}",
                                f"{s['avg_pts']:.1f}", line(s, "points"), f"#{s['vp']['rank']} of {s['n']}"]
                               for i, s in enumerate(scorers, 1)],
                              ["#", "Player", "Opp", "Proj. pts", "Season avg", "Prop line", f"Opp D vs pos."],
                              [W * 0.05, W * 0.3, W * 0.1, W * 0.12, W * 0.13, W * 0.12, W * 0.18]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Opp D vs pos.: where the opponent ranks in points allowed to the player's position "
                           f"(1 = fewest, {len(dash['teams'])} = most). Prop lines appear once the late-afternoon "
                           "update reads them.", S["small"]))

    # per-game sections
    story.append(PageBreak())
    for i, (g, dg, plist, spots) in enumerate(sections):
        blk = [game_header(g, dash, dg, lg, W), Spacer(1, 5), projection_strip(g, W), Spacer(1, 4)]
        sp, _ = spots_block(spots, cfg["spots_per_game"], W)
        if (g.get("spots")):
            sp.insert(0, Paragraph("<b>Schedule notes:</b> " + "; ".join(g["spots"]), S["small"]))
        if cfg["compact"]:
            left = [out_line(g, plist), *sp]
            tb = Table([[left, stat_table(g, dash, ranks, W * 0.4, compact=True)]], colWidths=[W * 0.58, W * 0.42])
            tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            blk += [tb, Spacer(1, 14)]
            story.append(KeepTogether(blk))
        else:
            left = [out_line(g, plist), *sp]
            tb = Table([[left, stat_table(g, dash, ranks, W * 0.4)]], colWidths=[W * 0.58, W * 0.42])
            tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            blk += [tb, Spacer(1, 6)]
            pt = player_table(g, plist, cfg["player_table"], W)
            if pt is not None:
                blk += [Paragraph("Projected box score <font size=7 color='#5b6470'>(small grey = season average, "
                                  "blue L = prop line)</font>", S["h3"]), pt]
            blk.append(Spacer(1, 16))
            story.append(KeepTogether(blk))
            story.append(CondPageBreak(3.6 * inch))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    shutil.copyfile(path, out_dir / "latest.pdf")
    idx_path = Path(out_dir).parent / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    days = sorted(set(idx.get(lg, []) + [day]), reverse=True)
    idx[lg] = days
    idx_path.write_text(json.dumps(idx, indent=1) + "\n")
    print(f"{lg}: {len(games)} games -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", choices=list(LEAGUES), action="append")
    ap.add_argument("--date")
    ap.add_argument("--dash", help="dashboard data.json path or URL (one league only)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    for lg in a.league or list(LEAGUES):
        build(lg, a.date, a.dash, Path(a.out))


if __name__ == "__main__":
    main()
