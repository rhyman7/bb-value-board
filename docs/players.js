// NBA player projections page: tonight's projections from the feed, the nightly grades, and the
// backtest. Projections are not picks: there are no prop lines here.
(function () {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const f1 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(1));
  const f2 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(2));
  const STATS = [["minutes", "Min"], ["points", "Pts"], ["rebounds", "Reb"], ["assists", "Ast"], ["threes", "3PM"]];
  async function get(url) {
    try { const r = await fetch(url, { cache: "no-cache" }); return r.ok ? await r.json() : null; } catch (e) { return null; }
  }
  function dayLabel(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return new Date(y, m - 1, d).toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
  }

  function drawTonight(feed) {
    const box = $("#tonight-body");
    if (!feed || !feed.games) { box.innerHTML = '<p class="empty">The projections feed could not be loaded.</p>'; return; }
    $("#stamp").textContent = "Updated " + new Date(feed.generatedAt).toLocaleString("en-US", { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
    const rows = [];
    for (const g of feed.games) {
      if (!g.players) continue;
      for (const side of ["away", "home"]) for (const p of g.players[side] || []) rows.push({ ...p, date: g.date, gid: g.id, site: side === "home" ? "vs" : "at" });
    }
    if (!rows.length) { box.innerHTML = '<p class="empty">No player projections in this update.</p>'; return; }
    const days = [...new Set(rows.map((r) => r.date))].sort();
    const st = { day: days[0], q: "", key: "points", asc: false, all: false };
    $("#counts").textContent = `${rows.length} players over ${days.length} day${days.length === 1 ? "" : "s"} · ` +
      (feed.injuryReport ? "injury report read" : "injury report not read, so full rosters are assumed");
    const seg = $("#day-seg");
    seg.innerHTML = days.map((d) => `<button type="button" data-day="${d}">${dayLabel(d)}</button>`).join("");
    const head = `<thead><tr><th class="l"><button class="sort" data-k="name">Player</button></th><th class="l"><button class="sort" data-k="team">Game</button></th>` +
      STATS.map(([k, lab]) => `<th><button class="sort" data-k="${k}">${lab}</button></th>`).join("") + "</tr></thead>";
    function render() {
      seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.day === st.day)));
      let list = rows.filter((r) => r.date === st.day);
      if (st.q) { const t = st.q.toLowerCase(); list = list.filter((r) => r.name.toLowerCase().includes(t) || r.team.toLowerCase() === t || r.opp.toLowerCase() === t); }
      const k = st.key;
      list.sort((a, b) => { const x = a[k], y = b[k]; const c = typeof x === "string" ? x.localeCompare(y) : x - y; return st.asc ? c : -c; });
      const more = !st.all && list.length > 50 ? list.length - 50 : 0;
      if (more) list = list.slice(0, 50);
      const body = list.map((r) => `<tr><td class="l">${esc(r.name)}</td><td class="l"><a href="game.html?id=${encodeURIComponent(r.gid)}">${esc(r.team)} ${r.site} ${esc(r.opp)}</a></td>` +
        STATS.map(([s]) => `<td>${f1(r[s])}${r.avg && r.avg[s] != null ? `<small>${f1(r.avg[s])}</small>` : ""}</td>`).join("") + "</tr>").join("");
      box.innerHTML = list.length ? `<div class="tbl-wrap"><table class="pl">${head}<tbody>${body}</tbody></table></div>${more ? `<p><button type="button" class="more" id="show-all">Show ${more} more players</button></p>` : ""}` : '<p class="empty">No players match.</p>';
      box.querySelectorAll("th").forEach((th) => { const b = th.querySelector("button"); if (b && b.dataset.k === st.key) th.setAttribute("aria-sort", st.asc ? "ascending" : "descending"); });
    }
    seg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { st.day = b.dataset.day; render(); } });
    $("#q").addEventListener("input", (e) => { st.q = e.target.value.trim(); render(); });
    box.addEventListener("click", (e) => {
      if (e.target.closest("#show-all")) { st.all = true; render(); return; }
      const b = e.target.closest("button.sort"); if (!b) return;
      if (st.key === b.dataset.k) st.asc = !st.asc; else { st.key = b.dataset.k; st.asc = b.dataset.k === "name" || b.dataset.k === "team"; }
      render();
    });
    render();
  }

  function gradeTable(title, g) {
    if (!g || !g.player_games) return "";
    const rows = STATS.map(([k, lab]) => { const x = g[k]; const better = x.model_mae < x.season_avg_mae;
      return `<tr><td>${lab}</td><td class="${better ? "w" : ""}">${f2(x.model_mae)}</td><td>${f2(x.season_avg_mae)}</td><td>${f1(x.model_closer_pct)}%</td></tr>`; }).join("");
    return `<div><h3>${esc(title)} · ${g.player_games} players</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Stat</th><th>Projection missed by</th><th>Season avg missed by</th><th>Closer</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  function drawGraded(gr) {
    const box = $("#graded-body");
    if (!gr || !gr.nights) { box.innerHTML = '<p class="empty">The first grades come the morning after the first logged night. Projections are saved before tip-off each day.</p>'; return; }
    const top = (gr.last_night_top || []).map((r) => `<tr><td class="l">${esc(r.name)}</td><td class="l">${esc(r.team)} vs ${esc(r.opp)}</td>
      <td>${f1(r.p_points)}</td><td class="w">${f1(r.a_points)}</td><td>${f1(r.p_rebounds)}</td><td class="w">${f1(r.a_rebounds)}</td>
      <td>${f1(r.p_assists)}</td><td class="w">${f1(r.a_assists)}</td><td>${f1(r.p_threes)}</td><td class="w">${f1(r.a_threes)}</td></tr>`).join("");
    box.innerHTML = `<div class="grades">${gradeTable("Last night (" + dayLabel(gr.lastNight) + ")", gr.last_night)}${gradeTable("Last 7 days", gr.last_7_days)}${gradeTable(`Season (${gr.nights} night${gr.nights === 1 ? "" : "s"})`, gr.season)}</div>
      ${top ? `<h3 style="margin-top:20px">Last night's top projected scorers: projection, then result</h3><div class="tbl-wrap"><table>
      <thead><tr><th class="l">Player</th><th class="l">Game</th><th colspan="2">Pts (proj · actual)</th><th colspan="2">Reb</th><th colspan="2">Ast</th><th colspan="2">3PM</th></tr></thead><tbody>${top}</tbody></table></div>` : ""}`;
  }

  function drawBacktest(bt) {
    const box = $("#bt-body");
    if (!bt || !bt.holdout) { box.innerHTML = '<p class="empty">The backtest file could not be loaded.</p>'; return; }
    const h = bt.holdout;
    $("#cv-model").textContent = f2(h.points.model_mae) + " points";
    $("#cv-avg").textContent = f2(h.points.season_avg_mae);
    $("#bt-note").textContent = `Every regular-season game ${bt.holdout_seasons}, using only what was known before tip-off; settings were fitted on ${bt.tune_seasons}. ` +
      `Players who played and average 15+ minutes when they play (${h.player_games.toLocaleString()} player-games). In the backtest the model knows which regulars sat, the way it would from the injury report.`;
    const rows = STATS.map(([k, lab]) => { const x = h[k];
      return `<tr><td>${lab}</td><td class="w">${f2(x.model_mae)}</td><td>${f2(x.season_avg_mae)}</td><td>${x.gain != null ? f2(x.gain) + " ± " + (x.gain_se < 0.005 ? "<0.01" : f2(x.gain_se)) : "–"}</td><td>${f1((100 * (x.season_avg_mae - x.model_mae)) / x.season_avg_mae)}%</td></tr>`; }).join("");
    const t = bt.teammates_out;
    const rows2 = t ? STATS.map(([k, lab]) => `<tr><td>${lab}</td><td class="w">${f2(t[k].model_mae)}</td><td>${f2(t[k].season_avg_mae)}</td></tr>`).join("") : "";
    box.innerHTML = `<div class="two"><div><h3>All player-games, ${esc(bt.holdout_seasons)}</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Stat</th><th>Projection missed by</th><th>Season avg missed by</th><th>Gain</th><th>Better by</th></tr></thead><tbody>${rows}</tbody></table></div></div>
      ${t ? `<div><h3>With a regular teammate out (${t.player_games.toLocaleString()} player-games)</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Stat</th><th>Projection missed by</th><th>Season avg missed by</th></tr></thead><tbody>${rows2}</tbody></table></div>
      <p class="note">This is where the projections earn most of their keep: the season average does not know who is missing.</p></div>` : ""}</div>`;
  }

  (async function boot() {
    const [feed, bt] = await Promise.all([get("../feed/nba.json"), get("players_backtest.json")]);
    drawTonight(feed);
    drawGraded(feed && feed.playerGrades);
    drawBacktest(bt);
  })();
})();
