// Past nights: each night's games as the board had them before tip-off, with the final and the
// result against the line the board showed. LEAGUE is set by the page.
(function () {
  const L = window.LEAGUE;
  const BREAK_EVEN = 52.4;
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const f1 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(1));
  const minus = (s) => String(s).replace(/-/g, "−");
  async function get(url) {
    try { const r = await fetch(url, { cache: "no-cache" }); return r.ok ? await r.json() : null; } catch (e) { return null; }
  }
  function dayLabel(iso, long) {
    const [y, m, d] = iso.split("-").map(Number);
    return new Date(y, m - 1, d).toLocaleDateString("en-US", long ? { weekday: "long", month: "long", day: "numeric" } : { weekday: "short", month: "short", day: "numeric" });
  }
  function spreadText(m, g) {
    if (m == null) return "–";
    if (Math.abs(m) < 0.05) return "Pick";
    return m > 0 ? `${esc(g.home)} ${minus("-" + f1(m))}` : `${esc(g.away)} ${minus("-" + f1(-m))}`;
  }
  const rec = (r) => (r && r.w + r.l + r.p ? `${r.w}-${r.l}${r.p ? "-" + r.p : ""}` : "–");
  const pct = (r, be) => (r && r.pct != null ? `<span class="${be && r.pct < BREAK_EVEN ? "below" : be ? "above" : ""}">${r.pct.toFixed(1)}%</span>` : "");
  const tag = (x) => (x ? `<span class="res ${x === "W" ? "w" : x === "L" ? "l" : "p"}">${x}</span>` : "–");

  function tiles(s) {
    return `<div class="tiles">
      <div class="tile"><span class="lab">Against the board's spread</span><b>${rec(s.ats)}</b><span class="vs">${pct(s.ats, true)} · 52.4% breaks even</span></div>
      <div class="tile"><span class="lab">Totals</span><b>${rec(s.ou)}</b><span class="vs">${pct(s.ou, true)}</span></div>
      <div class="tile"><span class="lab">Winners</span><b>${rec(s.su)}</b><span class="vs">${pct(s.su, false)}</span></div>
      <div class="tile"><span class="lab">Missed the margin by</span><b>${f1(s.model_mae)}</b><span class="vs">line: ${f1(s.line_mae)} (${s.with_line} games)</span></div>
    </div>`;
  }

  async function drawNight(day) {
    const box = $("#night-body");
    const n = await get(`../archive/${L}/${day}.json`);
    if (!n) { box.innerHTML = '<p class="empty">That night could not be loaded.</p>'; return; }
    const gs = Object.values(n.games).sort((a, b) => (a.tip || "").localeCompare(b.tip || ""));
    const rows = gs.map((g) => `<tr><td class="l team">${esc(g.away)} at ${esc(g.home)}</td>
      <td>${spreadText(g.margin, g)}</td><td>${spreadText(g.mkt_spread, g)}</td>
      <td class="team">${g.final ? `${g.final.away}-${g.final.home}` : '<span class="dim">pending</span>'}</td><td>${tag(g.ats)}</td>
      <td>${f1(g.total)}</td><td>${f1(g.mkt_total)}</td><td>${g.final ? g.final.away + g.final.home : "–"}</td><td>${tag(g.ou)}</td></tr>`).join("");
    box.innerHTML = `<h3>${dayLabel(day, true)}</h3><div class="tbl-wrap"><table>
      <thead><tr><th class="l">Game</th><th>Model</th><th>Line</th><th>Final</th><th>ATS</th><th>Model total</th><th>Line total</th><th>Final total</th><th>O/U</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="note">Numbers as the board had them at its last update before tip-off. ATS and O/U grade the model's side of the line shown, so a model number with no line is not graded.${L === "nba" ? ' Player projections for the same nights are graded on the <a href="players.html#graded">Players page</a>.' : ""}</p>`;
  }

  (async function boot() {
    const idx = await get(`../archive/${L}/index.json`);
    if (!idx || !idx.nights || !idx.nights.length) {
      $("#season-body").innerHTML = '<p class="empty">The archive starts with the first night the board is saved before tip-off. Check back tomorrow.</p>';
      return;
    }
    $("#stamp").textContent = "Updated " + new Date(idx.generatedAt).toLocaleString("en-US", { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
    $("#season-body").innerHTML = tiles(idx.season) + `<p class="note">${idx.season.graded} graded games over ${idx.nights.length} night${idx.nights.length === 1 ? "" : "s"}. The model trails the closing line in the backtest, so a winning stretch here is most likely noise until it holds up over several hundred games.</p>`;
    const list = idx.nights.map((n) => `<tr data-day="${n.date}"><td class="l"><button type="button" class="linkish" data-day="${n.date}">${dayLabel(n.date)}</button></td>
      <td>${n.graded}/${n.games}</td><td>${rec(n.ats)}</td><td>${rec(n.ou)}</td><td>${rec(n.su)}</td><td>${f1(n.model_mae)}</td><td>${f1(n.line_mae)}</td></tr>`).join("");
    $("#nights-body").innerHTML = `<div class="tbl-wrap"><table><thead><tr><th class="l">Night</th><th>Graded</th><th>ATS</th><th>O/U</th><th>Winners</th><th>Model miss</th><th>Line miss</th></tr></thead><tbody>${list}</tbody></table></div>`;
    let cur = location.hash.slice(1);
    if (!idx.nights.some((n) => n.date === cur)) cur = (idx.nights.find((n) => n.graded) || idx.nights[0]).date;
    const pick = (d) => { document.querySelectorAll("#nights-body tr[data-day]").forEach((tr) => tr.classList.toggle("on", tr.dataset.day === d)); drawNight(d); };
    $("#nights-body").addEventListener("click", (e) => { const b = e.target.closest("button[data-day]"); if (b) { history.replaceState(null, "", "#" + b.dataset.day); pick(b.dataset.day); $("#night").scrollIntoView({ behavior: "smooth" }); } });
    pick(cur);
  })();
})();
