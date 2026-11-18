// Basketball value board: reads the model feed, the backtest and the opening-line test and draws
// the page. LEAGUE ("nba" or "cbb") is set by the page. Nothing here is a bet signal: the backtests
// show the model trailing the closing line, and the page says so before anything else.
(function () {
  const L = window.LEAGUE;
  const BREAK_EVEN = 52.4;           // win rate needed at -110
  const $ = (s, el) => (el || document).querySelector(s);
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const f1 = (x) => (x == null || isNaN(x) ? "–" : Number(x).toFixed(1));
  const minus = (s) => String(s).replace(/-/g, "−");

  async function get(url) {
    try {
      const r = await fetch(url, { cache: "no-cache" });
      return r.ok ? await r.json() : null;
    } catch (e) {
      return null;
    }
  }

  function dayLabel(iso, long) {
    const [y, m, d] = iso.split("-").map(Number);
    const dt = new Date(y, m - 1, d);
    return dt.toLocaleDateString("en-US", long ? { weekday: "long", month: "long", day: "numeric" } : { weekday: "short", month: "short", day: "numeric" });
  }

  // margins are home minus away (+ = home favored), the same sign as the market spread in the feed
  function spreadText(m, g) {
    if (m == null) return "–";
    if (Math.abs(m) < 0.05) return "Pick";
    return m > 0 ? `${esc(g.home)} ${minus("-" + f1(m))}` : `${esc(g.away)} ${minus("-" + f1(-m))}`;
  }

  function gapText(g) {
    const parts = [];
    if (g.mkt_spread != null) {
      const d = g.margin - g.mkt_spread;
      if (Math.abs(d) >= 0.05) parts.push(`spread ${f1(Math.abs(d))} pts toward ${esc(d > 0 ? g.home : g.away)}`);
      else parts.push("spread matches the market");
    }
    if (g.mkt_total != null) {
      const d = g.total - g.mkt_total;
      parts.push(Math.abs(d) >= 0.05 ? `total ${f1(Math.abs(d))} ${d > 0 ? "higher" : "lower"}` : "total matches");
    }
    return parts.length ? "Model vs market: " + parts.join(", ") : "No market line yet";
  }

  const american = (p) => (p >= 0.5 ? Math.round((-100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p));
  const amText = (v) => (v > 0 ? "+" + v : minus(String(v)));
  // the favorite's price: the market's moneyline, and the model's fair price for the same team
  function favHome(g) {
    if (g.mkt_ml_home != null && g.mkt_ml_away != null) return g.mkt_ml_home <= g.mkt_ml_away;
    return g.win_prob >= 0.5;
  }
  function mlText(g) {
    if (g.mkt_ml_home == null || g.mkt_ml_away == null) return "–";
    return favHome(g) ? `${esc(g.home)} ${amText(g.mkt_ml_home)}` : `${esc(g.away)} ${amText(g.mkt_ml_away)}`;
  }
  function fairText(g) {
    const h = favHome(g), p = Math.min(0.995, Math.max(0.005, h ? g.win_prob : 1 - g.win_prob));
    return `${esc(h ? g.home : g.away)} ${amText(american(p))}`;
  }

  function gapSize(g) {
    return g.mkt_spread == null ? -1 : Math.abs(g.margin - g.mkt_spread);
  }

  function card(g) {
    const ph = g.win_prob, pa = 1 - ph;
    const tags = [];
    if (g.neutral) tags.push('<span class="tag">Neutral site</span>');
    const team = (code, p, rest, fav) => `
      <div class="bug-team${fav ? " fav" : ""}">
        <span class="code">${esc(code)}</span>
        <span class="rec">${rest != null ? (rest === 0 ? "Back-to-back" : rest + (rest === 1 ? " day" : " days") + " rest") : ""}</span>
        <span class="win" title="Model win probability">${Math.round(100 * p)}%</span>
      </div>`;
    const foot = [`<span>${gapText(g)}</span>`];
    if (g.missing) {
      const out = [];
      if (g.missing.away_names && g.missing.away_names.length) out.push(`${esc(g.away)}: ${g.missing.away_names.map(esc).join(", ")}`);
      if (g.missing.home_names && g.missing.home_names.length) out.push(`${esc(g.home)}: ${g.missing.home_names.map(esc).join(", ")}`);
      if (out.length) foot.push(`<span class="out">Out: ${out.join("; ")}</span>`);
    }
    if (g.spots && g.spots.length) foot.push(`<ul class="spots" aria-label="Situational spots">${g.spots.map((t) => `<li>${esc(t)}</li>`).join("")}</ul>`);
    return `
    <a class="bug" href="game.html?id=${encodeURIComponent(g.id)}">
      <div class="bug-top"><span>${dayLabel(g.date)}</span><span>${tags.join(" ")}</span></div>
      ${team(g.away, pa, g.rest && g.rest.away, pa > ph)}
      ${team(g.home, ph, g.rest && g.rest.home, ph >= pa)}
      <div class="bug-lines">
        <span class="h"></span><span class="h r">Market</span><span class="h r">Model</span>
        <span>Spread</span><span class="r b">${spreadText(g.mkt_spread, g)}</span><span class="r">${spreadText(g.margin, g)}</span>
        <span>Total</span><span class="r b">${g.mkt_total == null ? "–" : f1(g.mkt_total)}</span><span class="r">${f1(g.total)}</span>
        <span>Moneyline</span><span class="r b">${mlText(g)}</span><span class="r">${fairText(g)}</span>
      </div>
      <div class="bug-foot">${foot.join("")}<span class="go">Game page &rarr;</span></div>
    </a>`;
  }

  function drawGames(feed) {
    const box = $("#games-body");
    if (!feed || !feed.games) {
      box.innerHTML = '<p class="empty">The projections feed could not be loaded. It updates twice a day during the season.</p>';
      return;
    }
    const games = feed.games;
    const days = [...new Set(games.map((g) => g.date))].sort();
    // open on the first day with a game still to tip off (the late update runs after some games start)
    const now = Date.now();
    const upcomingDay = days.find((d) => games.some((g) => g.date === d && (!g.tip || Date.parse(g.tip) > now)));
    const state = { day: upcomingDay || days[days.length - 1] || null, sort: "time", q: "" };
    const withLine = games.filter((g) => g.mkt_spread != null).length;
    const notes = [`${games.length} games over ${days.length} day${days.length === 1 ? "" : "s"}`, `${withLine} with a market line`];
    if (L === "nba") notes.push(feed.injuryReport ? "injury report read" : "injury report not read, so full rosters are assumed");
    $("#games-counts").textContent = notes.join(" · ");
    $("#stamp").textContent = "Updated " + new Date(feed.generatedAt).toLocaleString("en-US", { weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });

    const daySeg = $("#day-seg");
    daySeg.innerHTML = days.map((d) => `<button type="button" data-day="${d}">${dayLabel(d)}</button>`).join("") + '<button type="button" data-day="">All</button>';
    const sortSeg = $("#sort-seg");
    const q = $("#team-q");

    function render() {
      daySeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String((b.dataset.day || null) === state.day)));
      sortSeg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.sort === state.sort)));
      let list = games.filter((g) => !state.day || g.date === state.day);
      if (state.q) {
        const t = state.q.toUpperCase();
        list = list.filter((g) => g.home.toUpperCase().includes(t) || g.away.toUpperCase().includes(t));
      }
      if (state.sort === "gap") list = [...list].sort((a, b) => gapSize(b) - gapSize(a));
      box.innerHTML = list.length ? `<div class="cards">${list.map(card).join("")}</div>` : '<p class="empty">No games match.</p>';
    }
    daySeg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { state.day = b.dataset.day || null; render(); } });
    sortSeg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { state.sort = b.dataset.sort; render(); } });
    if (q) q.addEventListener("input", () => { state.q = q.value.trim(); render(); });
    render();
  }

  const rec = (r) => (r && r.pct != null ? `${r.w}-${r.l}${r.p ? "-" + r.p : ""}` : "–");
  const pctCell = (r) => (r && r.pct != null ? `<span class="${r.pct >= BREAK_EVEN ? "above" : "below"}">${r.pct.toFixed(1)}%</span>` : "–");
  const vsBE = (r) => (r && r.pct != null ? `${r.pct >= BREAK_EVEN ? "+" : minus("-")}${Math.abs(r.pct - BREAK_EVEN).toFixed(1)}` : "–");

  function gapTable(title, buckets, unit) {
    const rows = Object.entries(buckets || {}).map(([k, r]) => `
      <tr><td>${esc(k.replace("+", ""))}+ ${unit}</td><td>${rec(r)}</td><td>${pctCell(r)}</td><td>${r && r.se != null ? "±" + r.se.toFixed(1) : "–"}</td><td>${vsBE(r)}</td></tr>`).join("");
    return `<div><h3>${title}</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Model off the line by</th><th>W-L-P</th><th>Win %</th><th>Std. error</th><th>vs ${BREAK_EVEN}%</th></tr></thead>
      <tbody>${rows}</tbody></table></div></div>`;
  }

  function tile(big, vs, lab) {
    return `<div class="tile"><b>${big}</b><span class="vs">${vs}</span><span class="lab">${lab}</span></div>`;
  }

  function drawRecordNBA(bt) {
    const box = $("#model-body");
    if (!bt) { box.innerHTML = '<p class="empty">The backtest summary could not be loaded.</p>'; return; }
    const ae = bt.ats_every_game, te = bt.totals_every_game;
    $("#model-lede").innerHTML = `Every game from ${esc(bt.seasons)} (${bt.games.toLocaleString()} games), each projected only from what was known before tip-off, then graded against the consensus closing line.`;
    const season = (bt.by_season || []).map((s) => `<tr><td>${s.season - 1}-${String(s.season).slice(2)}</td><td>${s.games.toLocaleString()}</td><td>${f1(s.model)}</td><td>${f1(s.close)}</td><td>${rec(s.ats)}</td><td>${pctCell(s.ats)}</td></tr>`).join("");
    box.innerHTML = `
      <div class="tiles">
        ${tile(f1(bt.margin_mae.model), `vs ${f1(bt.margin_mae.close)} for the closing line`, "Average miss on the final margin, points")}
        ${tile(f1(bt.total_mae.model), `vs ${f1(bt.total_mae.close)} for the closing total`, "Average miss on the total, points")}
        ${tile(`<span class="below">${ae.pct.toFixed(1)}%</span>`, `${rec(ae)} against the spread`, `Taking the model's side every game at the close (${BREAK_EVEN}% breaks even)`)}
        ${tile(`<span class="below">${te.pct.toFixed(1)}%</span>`, `${rec(te)} on totals`, "Taking the model's over or under every game")}
      </div>
      <div class="two">
        ${gapTable("Spreads, by how far the model was off the close", bt.ats_by_gap, "pts")}
        ${gapTable("Totals, by how far the model was off the close", bt.totals_by_gap, "pts")}
      </div>
      <div><h3>By season</h3><div class="tbl-wrap"><table>
        <thead><tr><th>Season</th><th>Games</th><th>Model miss</th><th>Close miss</th><th>ATS W-L-P</th><th>Win %</th></tr></thead>
        <tbody>${season}</tbody></table></div></div>
      <p class="note">The market weight is how much of the gap between the model and the close is worth taking: ${minus(bt.market_weight.side.toFixed(2))} on spreads (±${bt.market_weight.side_se.toFixed(2)}) and ${minus(bt.market_weight.total.toFixed(2))} on totals (±${bt.market_weight.total_se.toFixed(2)}). Zero or below means the best guess is the market's number on its own.</p>`;
  }

  // Ryan's situational spots, each tested against the line on its own
  function spotTable(sp) {
    if (!sp) return "";
    const cell = (t) => (t && t.pts != null ? `${t.pts > 0 ? "+" : ""}${minus(t.pts.toFixed(2))} <span class="muted">±${t.se.toFixed(2)}</span>` : "–");
    const ats = (t) => (t && t.ats ? pctCell(t.ats) + ` <span class="muted">(${t.games.toLocaleString()})</span>` : "–");
    const rows = Object.values(sp).map((v) => `<tr><td>${esc(v.label)}</td><td>${cell(v.tune)}</td><td>${cell(v.holdout)}</td><td>${ats(v.holdout)}</td><td>${cell(v.this_season)}</td></tr>`).join("");
    return `<div><h3>Situational spots against the line</h3><div class="tbl-wrap"><table>
      <thead><tr><th>Spot</th><th>Pts vs line, 2016-18</th><th>Pts vs line, 2019-23</th><th>Taking the spot's side, 2019-23</th><th>This season</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="note">Points each spot was worth against ESPN's spread in the direction the idea says (positive means it worked, for example the team in a letdown spot fell short of the line), with the standard error. To be worth adding, a spot has to move the result the same way in both periods by more than its standard error. Most flip sign between the two periods. Long road trips (2+ time zones) point the same way both times, by about half a point, but that is within about one standard error, and taking that side won 52.4%, exactly break-even. So the line mostly prices these spots. Long road trips are in the model as a small adjustment (about 1.3 points against the traveling team) and are being watched; the rest are listed on the game cards for context only. Tempo and opponent-adjusted efficiency are already the base of the ratings.</p></div>`;
  }

  // moneylines: the backtest against the closing spread's win probability, plus real prices once logged
  function mlSection(ml) {
    const x = ml && ml[L];
    if (!x) return "";
    const a = x.accuracy;
    const edges = Object.entries(x.by_edge).map(([k, v]) => `<tr><td>${k === "0%+" ? "Every game" : esc(k.replace("+", "")) + "+ better"}</td><td>${v.with_hold.bets.toLocaleString()}</td><td>${v.with_hold.win_pct.toFixed(1)}%</td><td><span class="${v.with_hold.roi_pct > 0 ? "above" : "below"}">${v.with_hold.roi_pct > 0 ? "+" : ""}${minus(v.with_hold.roi_pct.toFixed(1))}%</span> <span class="muted">±${v.with_hold.roi_se.toFixed(1)}</span></td></tr>`).join("");
    const prices = Object.entries(x.by_price).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v.bets.toLocaleString()}</td><td>${v.win_pct.toFixed(1)}%</td><td>${amText(v.avg_price_american)}</td><td><span class="${v.roi_pct > 0 ? "above" : "below"}">${v.roi_pct > 0 ? "+" : ""}${minus(v.roi_pct.toFixed(1))}%</span> <span class="muted">±${v.roi_se.toFixed(1)}</span></td></tr>`).join("");
    return `<div class="sub"><h3>Moneylines</h3>
      <p class="note">${esc(x.seasons)}, ${x.games.toLocaleString()} games. The market's win probability is more accurate than the model's (Brier score ${a.p_mkt.brier.toFixed(3)} vs ${a.p_model.brier.toFixed(3)}; lower is better), and betting the model's side whenever it rates a team higher than the price does not make money once the book's hold is paid.</p>
      <div class="two">
        <div><div class="tbl-wrap"><table><thead><tr><th>Model's win % above the price by</th><th>Bets</th><th>Won</th><th>Return</th></tr></thead><tbody>${edges}</tbody></table></div></div>
        <div><div class="tbl-wrap"><table><thead><tr><th>Price, 4%+ edge</th><th>Bets</th><th>Won</th><th>Typical price</th><th>Return</th></tr></thead><tbody>${prices}</tbody></table></div></div>
      </div>
      <p class="note">hoopR keeps closing spreads but no moneylines, so the market price here is the win probability the closing spread implies, with a standard 4.5% hold. Real moneylines are logged from now on and graded in the opening-line test below.</p></div>`;
  }

  function drawRecordCBB(bt) {
    const box = $("#model-body");
    if (!bt) { box.innerHTML = '<p class="empty">The backtest summary could not be loaded.</p>'; return; }
    const h = bt.home_games, ts = bt.this_season_so_far;
    $("#model-lede").innerHTML = `Every Division I game with a home team from ${esc(bt.holdout_seasons)} (${h.games.toLocaleString()} games), each projected only from what was known before tip-off, then graded against ESPN's spread. ESPN has no spreads in the data for 2024 to 2026.`;
    const bs = Object.entries(bt.by_season || {}).map(([s, x]) => `<tr><td>${s - 1}-${String(s).slice(2)}</td><td>${x.games.toLocaleString()}</td><td>${f1(x.model_mae)}</td><td>${f1(x.line_mae)}</td><td>${rec(x.ats_every_game)}</td><td>${pctCell(x.ats_every_game)}</td></tr>`).join("");
    const tsRow = ts && ts.games ? `<tr><td>This season</td><td>${ts.games.toLocaleString()}</td><td>${f1(ts.model_mae)}</td><td>${f1(ts.line_mae)}</td><td>${rec(ts.ats_every_game)}</td><td>${pctCell(ts.ats_every_game)}</td></tr>` : "";
    const g6 = (h.ats_by_gap || {})["6+"];
    box.innerHTML = `
      <div class="tiles">
        ${tile(f1(h.model_mae), `vs ${f1(h.line_mae)} for the line`, "Average miss on the final margin, points")}
        ${tile(`<span class="below">${h.ats_every_game.pct.toFixed(1)}%</span>`, `${rec(h.ats_every_game)} against the spread`, `Taking the model's side every game (${BREAK_EVEN}% breaks even)`)}
        ${ts && ts.games ? tile(f1(ts.model_mae), `vs ${f1(ts.line_mae)} for the line, ${ts.games} games`, "This season so far, average miss") : ""}
      </div>
      <div class="two">
        ${gapTable("Spreads, by how far the model was off the line", h.ats_by_gap, "pts")}
        <div><h3>By season</h3><div class="tbl-wrap"><table>
          <thead><tr><th>Season</th><th>Games</th><th>Model miss</th><th>Line miss</th><th>ATS W-L-P</th><th>Win %</th></tr></thead>
          <tbody>${bs}${tsRow}</tbody></table></div></div>
      </div>
      ${spotTable(bt.spots)}
      <p class="note">${g6 ? `The biggest disagreements (6+ points) went ${rec(g6)}, ${g6.pct.toFixed(1)}%. That is above ${BREAK_EVEN}% by less than one standard error (±${g6.se.toFixed(1)}), so it reads as noise, and the other buckets sit at or below break-even. ` : ""}Neutral-site games are left out: ESPN's lines for them are sometimes listed for the wrong side, which makes any record against them look far better than it is.</p>`;
  }

  function drawRatings(feed) {
    const box = $("#ratings-body");
    if (!box) return;
    if (!feed || !feed.teams) { box.innerHTML = '<p class="empty">No ratings in the feed.</p>'; return; }
    const rows = Object.entries(feed.teams).sort((a, b) => b[1].net - a[1].net)
      .map(([t, r], i) => `<tr><td>${i + 1}</td><td class="l team">${esc(t)}</td><td>${f1(r.off)}</td><td>${f1(r.def)}</td><td>${r.net > 0 ? "+" : ""}${minus(f1(r.net))}</td><td>${f1(r.pace)}</td></tr>`).join("");
    box.innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Rank</th><th class="l">Team</th><th>Offense</th><th>Defense</th><th>Net</th><th>Pace</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="note">Points scored and allowed per 100 possessions against an average opponent on a neutral floor (lower defense is better), and possessions per 48 minutes. Home court is worth about ${f1(feed.homeCourt)} points.</p>`;
  }

  function drawOpeners(op) {
    const box = $("#openers-body");
    const x = op && op[L];
    if (!x || !x.games) {
      box.innerHTML = `<p class="empty">Lines have been saved since November 18, 2026, the first time and every update after, so the model can be graded at the opening line and against how the line moves. Results show here once games finish, and are read at 300+ games.</p>`;
      return;
    }
    const part = (title, s, unit) => {
      if (!s || !s.games) return "";
      const rows = Object.entries(s.by_gap || {}).map(([k, r]) => `<tr><td>${k === "0+" ? "All games" : esc(k.replace("+", "")) + "+ " + unit}</td><td>${rec(r.record_at_open)}</td><td>${pctCell(r.record_at_open)}</td><td>${r.line_moved_toward_model_pct == null ? "–" : r.line_moved_toward_model_pct.toFixed(1) + "%"}</td><td>${r.avg_move_toward_model == null ? "–" : minus(f1(r.avg_move_toward_model))}</td></tr>`).join("");
      return `<div><h3>${title}</h3><div class="tbl-wrap"><table>
        <thead><tr><th>Model off the opener by</th><th>W-L-P at open</th><th>Win %</th><th>Line moved toward model</th><th>Avg move</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
        <p class="note">Average miss: model ${f1(s.mae.model_at_open)}, opener ${f1(s.mae.open)}, close ${f1(s.mae.close)}.</p></div>`;
    };
    const m = x.moneyline;
    const mlLive = m && m.games && m.by_edge ? `<div><h3>Moneylines at the last logged price</h3><div class="tbl-wrap"><table>
        <thead><tr><th>Model's win % above the no-vig price by</th><th>Bets</th><th>Won</th><th>Return</th></tr></thead>
        <tbody>${Object.entries(m.by_edge).map(([k, v]) => `<tr><td>${k === "0%+" ? "Any" : esc(k)}</td><td>${v.bets}</td><td>${v.won}</td><td>${v.roi_pct == null ? "–" : minus(v.roi_pct.toFixed(1)) + "%"} <span class="muted">${v.roi_se == null ? "" : "±" + v.roi_se.toFixed(1)}</span></td></tr>`).join("")}</tbody></table></div></div>` : "";
    box.innerHTML = `${x.note ? `<p class="note"><strong>${esc(x.note)}.</strong></p>` : ""}
      <div class="two">${part("Spreads", x.side, "pts")}${part("Totals", x.total, "pts")}</div>
      ${mlLive}
      <p class="note">If the line keeps moving toward the model more than half the time, the model is seeing something before the market does. That is the first sign of an edge, and it shows up long before a win-loss record means anything.</p>`;
  }

  // NBA only: does the model beat the line when injury news breaks before tip (pretip.py)
  function drawPretip(pt) {
    const box = $("#openers-body");
    if (!box) return;
    const body = pt && pt.news_events ? `<div class="tbl-wrap"><table><thead><tr><th></th><th>Result</th></tr></thead><tbody>
        <tr><td>Times injury news moved the model 0.5+ points</td><td>${pt.news_events}</td></tr>
        <tr><td>Line then moved toward the model</td><td>${pt.line_moved_toward_model_pct == null ? "–" : pt.line_moved_toward_model_pct.toFixed(1) + "%"}</td></tr>
        <tr><td>Average move toward the model</td><td>${pt.avg_move_toward_model == null ? "–" : minus(f1(pt.avg_move_toward_model)) + " pts"}</td></tr>
        <tr><td>Model's side at the line when the news broke</td><td>${rec(pt.record_at_news_line)} ${pctCell(pt.record_at_news_line)}</td></tr>
        <tr><td>Same side at the last line before tip</td><td>${rec(pt.record_at_close_same_side)} ${pctCell(pt.record_at_close_same_side)}</td></tr></tbody></table></div>
        <p class="note">${pt.runs} pre-tip runs over ${pt.games_logged} games. Too early to read before a couple of hundred news events.</p>`
      : `<p class="empty">From November 18, 2026 the board re-reads the injury report every 30 minutes before tip-off, updates the model's number and logs the line. Results show here once news has moved the model and the games are over.</p>`;
    box.insertAdjacentHTML("beforeend", `<h3 style="margin-top:24px">When injury news breaks before tip</h3>
      <p class="note">If the model reacts to a late scratch before the books do, the line should keep moving its way after the news. This is where an edge is most likely to show.</p>${body}`);
  }

  // NBA only: tests on past lines (Odds API) and the live first-half record
  function drawSoft(past, tt, sm) {
    const box = $("#soft-body");
    if (!box) return;
    const r = (x) => x ? `${rec(x)}` : "–";
    const p = (x) => x && x.pct != null ? x.pct.toFixed(1) + "%" : "–";
    const fair = past && past.nba_without_injury_hindsight, t = tt && tt.nba_without_injury_hindsight;
    const pastRows = [
      fair ? `<tr><td>Spread at the morning line (2024-26)</td><td>${r(fair.side.vs_opener["0+"])}</td><td>${p(fair.side.vs_opener["0+"])}</td></tr>` : "",
      fair ? `<tr><td>Total at the morning line (2024-26)</td><td>${r(fair.total.vs_opener["0+"])}</td><td>${p(fair.total.vs_opener["0+"])}</td></tr>` : "",
      t ? `<tr><td>Team totals implied by the closing spread and total (2021-26)</td><td>${r(t.by_gap["0+"])}</td><td>${p(t.by_gap["0+"])}</td></tr>` : ""].join("");
    const live = (title, s, unit) => {
      if (!s || !s.games) return "";
      const rows = Object.entries(s.by_gap).map(([k, x]) => `<tr><td>${k === "0+" ? "Every game" : esc(k.replace("+", "")) + "+ " + unit}</td><td>${rec(x)}</td><td>${pctCell(x)}</td><td>${x.roi == null ? "–" : minus(x.roi.toFixed(1)) + "%"}</td></tr>`).join("");
      return `<div><h3>${title}</h3><div class="tbl-wrap"><table><thead><tr><th>Model off the line by</th><th>W-L-P</th><th>Win %</th><th>Return</th></tr></thead><tbody>${rows}</tbody></table></div>
        <p class="note">Average miss: model ${f1(s.mae.model)}, line ${f1(s.mae.line)}.</p></div>`;
    };
    const liveHtml = sm && sm.games ? `<div class="two">${live("First-half spreads", sm.h1_spread, "pts")}${live("First-half totals", sm.h1_total, "pts")}</div>
      <p class="note">${sm.games} games through ${esc(sm.last_night)}. Taking the model's side of the consensus first-half line at its median price, read on the late-afternoon update. Too early to read before a few hundred games.</p>`
      : `<p class="empty">First-half lines are read on the late-afternoon update from November 18, 2026 and graded the next morning. The record shows here once games finish.</p>`;
    box.innerHTML = `${pastRows ? `<h3>Tested on past lines: no edge</h3><div class="tbl-wrap"><table><thead><tr><th>Taking the model's side</th><th>W-L-P</th><th>Win %</th></tr></thead><tbody>${pastRows}</tbody></table></div>
      <p class="note">The model here leaves out who sat, which it can't know at the morning line. The morning line is a noon ET consensus, about 9 hours before tip. 52.4% breaks even at −110.</p>` : ""}
      <h3 style="margin-top:20px">First half, live</h3>${liveHtml}`;
  }

  // fill the warning's numbers from the latest backtest so it never goes stale
  function drawCaveat(bt) {
    const set = (id, v) => { const el = document.getElementById(id); if (el && v != null) el.textContent = v; };
    if (!bt) return;
    const m = L === "nba" ? bt.margin_mae : { model: bt.home_games.model_mae, close: bt.home_games.line_mae };
    const a = L === "nba" ? bt.ats_every_game : bt.home_games.ats_every_game;
    set("cv-model", f1(m.model)); set("cv-close", f1(m.close)); set("cv-ats", a.pct.toFixed(1) + "%");
  }

  async function boot() {
    const [feed, bt, op, ml] = await Promise.all([get(`../feed/${L}.json`), get(`../${L}/backtest.json`), get("../openers.json"), get("../moneyline.json")]);
    drawCaveat(bt);
    drawGames(feed);
    if (L === "nba") { drawRecordNBA(bt); drawRatings(feed); } else drawRecordCBB(bt);
    const mlHtml = mlSection(ml);
    if (mlHtml) $("#model-body").insertAdjacentHTML("beforeend", mlHtml);
    drawOpeners(op);
    if (L === "nba") {
      const [past, tt, sm, pt] = await Promise.all([get("../opening_lines_test.json"), get("../team_totals_test.json"), get("soft_markets.json"), get("pretip.json")]);
      drawSoft(past, tt, sm);
      drawPretip(pt);
    }
  }
  boot();
})();
