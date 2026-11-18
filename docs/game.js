// Game page: one game from the feed, with how the model's number is built, how the two teams
// match up on the four factors, and each team's last ten games. LEAGUE is set by the page.
(function () {
  const L = window.LEAGUE;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const f1 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(1));
  const minus = (s) => String(s).replace(/-/g, "−");
  const sgn = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
  const amText = (v) => (v > 0 ? "+" + v : minus(String(v)));
  const american = (p) => (p >= 0.5 ? Math.round((-100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p));

  function dayLabel(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return new Date(y, m - 1, d).toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric" });
  }
  function spreadText(m, g) {
    if (m == null) return "–";
    if (Math.abs(m) < 0.05) return "Pick";
    return m > 0 ? `${esc(g.home)} ${minus("-" + f1(m))}` : `${esc(g.away)} ${minus("-" + f1(-m))}`;
  }

  function contrib(pieces, total, unitNote, signed) {
    const max = Math.max(1, ...pieces.map((p) => Math.abs(p.pts)));
    const rows = pieces.map((p) => {
      const w = (100 * Math.abs(p.pts)) / max;
      const bar = signed
        ? `<span class="cbar"><i class="${p.pts >= 0 ? "pos" : "neg"}" style="${p.pts >= 0 ? "left:50%" : "right:50%"};width:${w / 2}%"></i></span>`
        : `<span class="cbar u"><i class="pos" style="left:0;width:${w}%"></i></span>`;
      return `<span class="lab">${esc(p.label)}</span>${bar}<span class="v">${signed ? sgn(p.pts) : f1(p.pts)}</span>`;
    }).join("");
    return `<div class="contrib">${rows}<span class="sep"></span><span class="lab sum">Model</span><span class="cap">${unitNote}</span><span class="v sum">${signed ? sgn(total) : f1(total)}</span></div>`;
  }

  function rk(r, n) {
    if (r == null) return "";
    const q = Math.min(5, 1 + Math.floor(((r - 1) * 5) / n));
    return `<span class="rk q${q}">${r}</span>`;
  }

  function matchTable(offTeam, defTeam, T, labels, nTeams) {
    const o = T[offTeam] && T[offTeam].stats, d = T[defTeam] && T[defTeam].stats;
    if (!o || !d) return "";
    const rows = Object.entries(labels).map(([k, lab]) => `<tr><td>${esc(lab)}</td>
        <td>${f1(o.off[k])} ${rk(o.ranks["off_" + k], nTeams)}</td><td>${f1(d.def[k])} ${rk(d.ranks["def_" + k], nTeams)}</td></tr>`).join("");
    return `<div><h3>${esc(offTeam)} offense vs ${esc(defTeam)} defense</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Stat</th><th>${esc(offTeam)} offense</th><th>${esc(defTeam)} allows</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  function formTable(team, list) {
    if (!list || !list.length) return "";
    const su = list.filter((x) => x.result === "W").length;
    const ats = list.filter((x) => x.ats), aw = ats.filter((x) => x.ats === "W").length, al = ats.filter((x) => x.ats === "L").length;
    const rows = list.map((x) => `<tr><td>${x.date.slice(5).replace("-", "/")}</td><td class="l">${esc(x.site)} ${esc(x.opp)}</td>
        <td class="${x.result === "W" ? "w" : "dim"}">${x.result} ${esc(x.score)}</td><td>${x.line == null ? "–" : x.line > 0 ? "+" + f1(x.line) : x.line === 0 ? "PK" : minus(f1(x.line))}</td><td>${x.ats || "–"}</td></tr>`).join("");
    return `<div><h3>${esc(team)}: ${su}-${list.length - su} last ${list.length}${ats.length ? `, ${aw}-${al} against the spread` : ""}</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Date</th><th class="l">Opponent</th><th>Result</th><th>Line</th><th>ATS</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  async function boot() {
    const id = new URLSearchParams(location.search).get("id");
    let feed = null;
    try { const r = await fetch(`../feed/${L}.json`, { cache: "no-cache" }); feed = r.ok ? await r.json() : null; } catch (e) { feed = null; }
    const g = feed && feed.games.find((x) => x.id === id);
    if (!g) { $("#page").innerHTML = '<p class="empty">This game is not on the board any more. <a href="./">Back to the board</a>.</p>'; return; }
    document.title = `${g.away} at ${g.home} · ${L === "nba" ? "NBA" : "College"} Value Board`;
    const T = feed.teams || {};
    const nTeams = feed.nTeams || Object.keys(T).length || 30;
    const ph = g.win_prob, pa = 1 - ph;
    const mlRow = g.mkt_ml_home != null && g.mkt_ml_away != null
      ? `<tr><td>Moneyline</td><td>${esc(g.away)} ${amText(g.mkt_ml_away)} · ${esc(g.home)} ${amText(g.mkt_ml_home)}</td><td>${esc(g.away)} ${amText(american(Math.min(0.995, Math.max(0.005, pa))))} · ${esc(g.home)} ${amText(american(Math.min(0.995, Math.max(0.005, ph))))}</td></tr>`
      : `<tr><td>Moneyline (fair)</td><td>–</td><td>${esc(g.away)} ${amText(american(Math.min(0.995, Math.max(0.005, pa))))} · ${esc(g.home)} ${amText(american(Math.min(0.995, Math.max(0.005, ph))))}</td></tr>`;
    const team = (code, p, rest, fav) => `<div class="gh-team${fav ? " fav" : ""}"><span class="code">${esc(code)}</span>
      <span class="rec">${T[code] && T[code].net != null ? `Net ${sgn(T[code].net)} per 100` : ""}${rest != null ? ` · ${rest === 0 ? "back-to-back" : rest + (rest === 1 ? " day" : " days") + " rest"}` : ""}</span>
      <span class="win">${Math.round(100 * p)}%</span></div>`;
    const labels = { efg: "Effective FG%", tov: "Turnover %", orb: "Off. rebound %", ftr: "FT rate (FTM per FGA)", tpr: "3-point rate", ortg: "Points per 100", pace: "Pace" };
    const b = g.build || { margin: [], total: [] };
    $("#page").innerHTML = `
      <p class="crumbs"><a href="./">&larr; All games</a><span>${dayLabel(g.date)}${g.neutral ? " · neutral site" : ""}</span></p>
      <div class="gh">${team(g.away, pa, g.rest && g.rest.away, pa > ph)}${team(g.home, ph, g.rest && g.rest.home, ph >= pa)}</div>
      <aside class="caveat slim"><p>The model trails the closing line (see the <a href="./#model">model record</a>). Its number is shown next to the market's for context, not as a bet.</p></aside>
      <section><div class="section-head"><h2>The numbers</h2></div><div class="tbl-wrap"><table>
        <thead><tr><th>Line</th><th>Market</th><th>Model</th></tr></thead><tbody>
        <tr><td>Spread</td><td>${spreadText(g.mkt_spread, g)}</td><td>${spreadText(g.margin, g)}</td></tr>
        <tr><td>Total</td><td>${g.mkt_total == null ? "–" : f1(g.mkt_total)}</td><td>${f1(g.total)}</td></tr>
        ${mlRow}</tbody></table></div></section>
      ${b.margin.length ? `<section><div class="section-head"><h2>How the spread is built</h2>
        <p class="note">Every input of the game model in points, toward ${esc(g.home)} (+) or ${esc(g.away)} (−). They add up to the model's margin.</p></div>
        ${contrib(b.margin, g.margin, `+ favors ${esc(g.home)}`, true)}</section>` : ""}
      ${b.total.length ? `<section><div class="section-head"><h2>How the total is built</h2></div>${contrib(b.total, g.total, "points", false)}</section>` : ""}
      ${T[g.home] && T[g.home].stats ? `<section><div class="section-head"><h2>How they match up</h2>
        <p class="note">Season to date. Ranks are out of ${nTeams} (1 = best for that side of the ball; pace and 3-point rate rank highest first).</p></div>
        <div class="two">${matchTable(g.away, g.home, T, labels, nTeams)}${matchTable(g.home, g.away, T, labels, nTeams)}</div></section>` : ""}
      ${T[g.home] && T[g.home].last10 ? `<section><div class="section-head"><h2>Recent form</h2>
        <p class="note">Lines are ESPN's, from the team's side (negative = favored).</p></div>
        <div class="two">${formTable(g.away, T[g.away] && T[g.away].last10)}${formTable(g.home, T[g.home] && T[g.home].last10)}</div></section>` : ""}`;
  }
  boot();
})();
