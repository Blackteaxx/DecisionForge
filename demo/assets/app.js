/* DecisionForge demo — hero, workflow, benchmark chart, DAG and trajectory player. */
(() => {
  "use strict";

  const language = new URLSearchParams(location.search).get('lang') === 'en' ? 'en' : 'zh';
  const C = window.DFORGE_COPY[language];
  const t = text => language === 'en' ? (window.DFORGE_UI[text] || text) : text;
  document.documentElement.lang = language === 'en' ? 'en' : 'zh-CN';
  document.querySelectorAll('[data-i18n]').forEach(el => el.innerHTML = t(el.dataset.i18n));
  const toggle = document.querySelector('#language-toggle');
  toggle.textContent = language === 'en' ? '中文' : 'English';
  toggle.onclick = () => { const url = new URL(location.href); url.searchParams.set('lang', language === 'en' ? 'zh' : 'en'); location.href = url; };
  const T = window.DFORGE_TRAJECTORIES || {};
  const FLOW = C.flow;
  const TRAJ_IDS = Object.keys(C.trajectories).filter((id) => T[id]);
  const REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const h = (tag, attrs = {}, html = "") => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "style") node.style.cssText = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    if (html) node.innerHTML = html;
    return node;
  };
  const fmtNum = (n) => Number(n).toLocaleString("en-US");
  const nodeLabel = (id) => (FLOW.nodes[id] || {}).label || id;

  const TZ_FMT = new Intl.DateTimeFormat("zh-CN", {
    timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false,
  });
  const fmtTime = (ts) => {
    const p = Object.fromEntries(TZ_FMT.formatToParts(new Date(ts)).map((x) => [x.type, x.value]));
    return `${p.month}-${p.day} ${p.hour}:${p.minute}`;
  };
  const fmtClock = (ts) => fmtTime(ts).slice(6);
  const fmtDur = (ms) => {
    const m = Math.round(ms / 60000);
    if (m < 60) return `${m} 分钟`;
    const hrs = ms / 3600000;
    if (hrs < 48) return `${hrs.toFixed(hrs < 10 ? 1 : 0)} 小时`;
    return `${(hrs / 24).toFixed(1)} 天`;
  };

  // ------------------------------------------------------------------ markdown
  function inline(s) {
    let out = esc(s);
    const codes = [];
    out = out.replace(/`([^`]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
    out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, text, url) => {
      if (/^https?:/.test(url)) return `<a href="${url}" target="_blank" rel="noopener">${text}</a>`;
      return `<span class="lpath" title="${url}">${text}</span>`;
    });
    out = out.replace(/(^|[\s(（])(https?:\/\/[^\s<)）]+)/g, (_, pre, url) => `${pre}<a href="${url}" target="_blank" rel="noopener">${url}</a>`);
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
    out = out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`);
    return out;
  }

  function progressBlock(code) {
    const lines = code.split("\n").filter((l) => l.trim());
    const items = lines.filter((l) => /^\s*\[[ x~!]\]/.test(l));
    if (items.length < 3) return null;
    const title = lines.find((l) => !/^\s*\[/.test(l)) || t("进度");
    const cls = { x: ["x", "✓"], "~": ["w", "◐"], "!": ["b", "!"], " ": ["o", "○"] };
    return `<div class="progress"><div class="ptitle">${esc(title)}</div>${items.map((l) => {
      const m = l.match(/^\s*\[([ x~!])\]\s*(.*)$/);
      const [c, icon] = cls[m[1]];
      return `<div class="pi ${c}"><i>${icon}</i><span>${inline(m[2])}</span></div>`;
    }).join("")}</div>`;
  }

  function md(src) {
    const lines = String(src || "").replace(/\r/g, "").split("\n");
    const out = [];
    let i = 0;
    const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
    const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
    while (i < lines.length) {
      const line = lines[i];
      if (/^\s*```/.test(line)) {
        const buf = [];
        i++;
        while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
        i++;
        const code = buf.join("\n");
        out.push(progressBlock(code) || `<pre><code>${esc(code)}</code></pre>`);
        continue;
      }
      if (/^\s*<\/?details>\s*$/.test(line)) { out.push(line.includes("/") ? "</details>" : "<details>"); i++; continue; }
      const sm = line.match(/^\s*<summary>(.*)<\/summary>\s*$/);
      if (sm) { out.push(`<summary>${inline(sm[1])}</summary>`); i++; continue; }
      if (/^\s*---+\s*$/.test(line)) { i++; continue; }
      if (/^\s*\|/.test(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
        out.push(`<table><thead><tr>${head.map((c) => `<th>${inline(c)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
        continue;
      }
      const hm = line.match(/^(#{1,4})\s+(.*)$/);
      if (hm) { out.push(`<h4>${inline(hm[2])}</h4>`); i++; continue; }
      if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
        const ordered = /^\s*\d+\./.test(line);
        const items = [];
        while (i < lines.length && (/^\s*([-*]|\d+\.)\s+/.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
          const l = lines[i++];
          if (/^\s*([-*]|\d+\.)\s+/.test(l)) items.push(l.replace(/^\s*([-*]|\d+\.)\s+/, ""));
          else items[items.length - 1] += " " + l.trim();
        }
        const tag = ordered ? "ol" : "ul";
        out.push(`<${tag}>${items.map((t) => `<li>${inline(t)}</li>`).join("")}</${tag}>`);
        continue;
      }
      if (/^\s*>\s?/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`);
        continue;
      }
      if (!line.trim()) { i++; continue; }
      const buf = [];
      while (i < lines.length && lines[i].trim() && !/^\s*(```|\||#{1,4}\s|[-*]\s|\d+\.\s|>|<\/?details>|<summary>|---+\s*$)/.test(lines[i])) buf.push(lines[i++]);
      if (!buf.length) buf.push(lines[i++]);
      out.push(`<p>${buf.map(inline).join("<br>")}</p>`);
    }
    return out.join("");
  }

  // ------------------------------------------------------------------ derived trajectory data
  const DERIVED = {};
  for (const id of TRAJ_IDS) {
    const events = T[id].events;
    const runs = [];
    events.forEach((ev, i) => {
      const last = runs[runs.length - 1];
      if (!last || last.n !== ev.n || last.r !== ev.r) runs.push({ n: ev.n, r: ev.r, start: i, end: i + 1 });
      else last.end = i + 1;
    });
    const nodeStats = {};
    events.forEach((ev, i) => {
      const s = (nodeStats[ev.n] ||= { first: {}, msgs: 0, tools: 0, asks: 0, rounds: new Set() });
      s.rounds.add(ev.r);
      if (s.first[ev.r] == null) s.first[ev.r] = i;
      if (ev.k === "tools") s.tools += ev.items.length;
      else if (ev.k === "user" || ev.k === "assistant") s.msgs += 1;
      else if (ev.k === "ask") s.asks += 1;
    });
    let msgs = 0, tools = 0;
    const cum = events.map((ev) => {
      if (ev.k === "tools") tools += ev.items.length;
      else if (ev.k === "user" || ev.k === "assistant" || ev.k === "status") msgs += 1;
      return { msgs, tools };
    });
    DERIVED[id] = { runs, nodeStats, cum, totalMsgs: msgs, totalTools: tools };
  }
  const sumStat = (f) => TRAJ_IDS.reduce((a, id) => a + f(T[id].stats), 0);

  // ------------------------------------------------------------------ top bar
  function initTopbar() {
    const bar = $("#topbar");
    const onScroll = () => bar.classList.toggle("solid", window.scrollY > 40);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  // ------------------------------------------------------------------ hero
  function renderHero() {
    $("#hero-eyebrow").textContent = C.hero.eyebrow;
    $("#hero-title").innerHTML = C.hero.title;
    $("#hero-lead").textContent = C.hero.lead;
    heroField($("#hero-canvas"));
  }

  /** Two drifting ribbons of ASCII glyphs — a light take on a text-art backdrop. */
  function heroField(cv) {
    const ctx = cv.getContext("2d");
    const RAMP = " .·:-=+*#%@";
    const BUCKETS = 9;
    let W = 0, H = 0, cols = 0, rows = 0, cw = 0, chh = 0, fs = 13;
    let mouse = { x: -1, y: -1 }, running = true, last = 0;

    function resize() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      W = cv.clientWidth; H = cv.clientHeight;
      cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      fs = W < 700 ? 10 : 12;
      cw = fs * 0.64; chh = fs * 1.2;
      cols = Math.ceil(W / cw); rows = Math.ceil(H / chh);
      ctx.font = `${fs}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textBaseline = "top";
    }
    const smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

    function draw(time) {
      const t = time * 0.001;
      ctx.clearRect(0, 0, W, H);
      const buckets = Array.from({ length: BUCKETS }, () => []);
      const accent = [];
      for (let c = 0; c < cols; c++) {
        const x = c / cols;
        const fade = smooth(0.2, 0.62, x);
        if (fade <= 0) continue;
        const r1 = 0.40 + 0.15 * Math.sin(x * 3.1 + t * 0.32) + 0.045 * Math.sin(x * 9.3 - t * 0.7);
        const r2 = 0.60 + 0.13 * Math.sin(x * 2.4 - t * 0.26 + 1.9) + 0.04 * Math.sin(x * 11.7 + t * 0.55);
        const w1 = 0.06 + 0.045 * (0.5 + 0.5 * Math.sin(x * 4.7 + t * 0.45));
        const w2 = 0.05 + 0.04 * (0.5 + 0.5 * Math.sin(x * 5.9 - t * 0.38 + 0.8));
        for (let r = 0; r < rows; r++) {
          const y = r / rows;
          const d1 = Math.exp(-(((y - r1) / w1) ** 2));
          const d2 = Math.exp(-(((y - r2) / w2) ** 2)) * 0.9;
          let v = Math.max(d1, d2);
          const grain = 0.5 + 0.5 * Math.sin(c * 0.71 + r * 1.37 + t * 1.6) * Math.sin(c * 0.23 - r * 0.41 - t * 0.9);
          v *= (0.55 + 0.45 * grain) * fade;
          if (mouse.x >= 0) {
            const dx = (c * cw - mouse.x) / 160, dy = (r * chh - mouse.y) / 160;
            v += 0.35 * Math.exp(-(dx * dx + dy * dy)) * fade;
          }
          if (v < 0.07) continue;
          v = Math.min(1, v);
          const ch = RAMP[Math.min(RAMP.length - 1, Math.floor(v * RAMP.length))];
          if (v > 0.72 && (c * 7 + r * 13) % 23 === 0) { accent.push([ch, c * cw, r * chh]); continue; }
          buckets[Math.min(BUCKETS - 1, Math.floor(v * BUCKETS))].push([ch, c * cw, r * chh]);
        }
      }
      buckets.forEach((list, b) => {
        if (!list.length) return;
        ctx.fillStyle = `rgba(17,17,19,${(0.08 + 0.5 * (b / (BUCKETS - 1))).toFixed(3)})`;
        for (const [ch, x, y] of list) ctx.fillText(ch, x, y);
      });
      ctx.fillStyle = "rgba(67,56,202,.75)";
      for (const [ch, x, y] of accent) ctx.fillText(ch, x, y);
    }

    function loop(time) {
      if (!running) return;
      if (time - last > 40) { draw(time); last = time; }
      requestAnimationFrame(loop);
    }
    resize();
    window.addEventListener("resize", () => { resize(); if (REDUCED) draw(0); });
    cv.parentElement.addEventListener("mousemove", (e) => { const r = cv.getBoundingClientRect(); mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; });
    cv.parentElement.addEventListener("mouseleave", () => { mouse = { x: -1, y: -1 }; });
    if (REDUCED) { draw(8000); return; }
    new IntersectionObserver(([en]) => {
      const vis = en.isIntersecting;
      if (vis && !running) { running = true; requestAnimationFrame(loop); }
      running = vis;
    }).observe(cv);
    requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ intro, workflow, gates
  function figure(value, unit, label) {
    return `<div><dd>${value}<small>${unit}</small></dd><dt>${label}</dt></div>`;
  }

  function renderIntro() {
    $("#intro-title").innerHTML = C.intro.title;
    $("#intro-body").textContent = C.intro.body;
    const benches = C.domains.reduce((a, d) => a + d.benchmarks.length, 0);
    $("#figures").innerHTML = [
      figure(TRAJ_IDS.length, t("次"), t("真实任务 · Codex 与 Claude Code")),
      figure(fmtNum(sumStat((s) => s.tools)), t("次"), t("agent 执行的命令与工具调用")),
      figure(sumStat((s) => s.asks), t("次"), t("关键决定由你确认")),
      figure(benches, t("项"), language === "en" ? `Benchmarks · ${C.domains.length} domains` : `基准评测 · ${C.domains.length} 个领域`),
    ].join("");
  }

  function renderSteps() {
    const box = $("#steps");
    for (const s of C.steps) {
      const el = h("article", { class: "step" }, `
        <div class="no"><span>${s.no}</span><span>${esc(s.skill)}</span></div>
        <h3>${esc(s.title)}</h3><p>${esc(s.body)}</p>`);
      const inst = h("div", { class: "inst" });
      for (const id of TRAJ_IDS) {
        inst.appendChild(h("button", { title: `在「${C.trajectories[id].tab}」回放中查看`, onclick: () => openTrajectory(id, s.node) },
          `<b>${esc(C.trajectories[id].tab.slice(0, 2))}</b><span>${esc(s[id])}</span><i>→</i>`));
      }
      el.appendChild(inst);
      box.appendChild(el);
    }
  }

  function renderGates() {
    $("#gates-title").innerHTML = C.gates.title;
    $("#gates-body").textContent = C.gates.body;
    const asks = [];
    for (const id of TRAJ_IDS) {
      T[id].events.forEach((ev, idx) => {
        if (ev.k !== "ask") return;
        ev.questions.forEach((q, qi) => {
          const ans = ev.answers.find((a) => a.q.slice(0, 30) === q.q.slice(0, 30)) || ev.answers[qi];
          if (ans) asks.push({ id, idx, t: ev.t, q: q.q, a: ans.a });
        });
      });
    }
    asks.sort((a, b) => (a.t < b.t ? -1 : 1));
    const list = $("#gate-list");
    for (const g of asks) {
      list.appendChild(h("li", { title: t("在回放中查看"), onclick: () => openTrajectory(g.id, null, null, g.idx) },
        `<span class="who"><b>${esc(C.trajectories[g.id].tab)}</b>${fmtTime(g.t)}</span><span class="q">${esc(g.q)}</span><span class="a">${esc(g.a)}</span><span class="go">→</span>`));
    }
  }

  // ------------------------------------------------------------------ benchmark chart
  function renderPerformance() {
    $("#perf-title").innerHTML = C.performance.title;
    $("#perf-body").textContent = C.performance.body;
    const methods = C.methods;
    const palette = window.DFORGE_METHOD_PALETTE;
    const darkMode = window.matchMedia("(prefers-color-scheme: dark)");
    const colorVar = (m) => `var(--m-${m.id})`;
    const paintVar = (m) => `var(--m-${m.id}-paint)`;
    const gradients = methods.filter((m) => palette.light[m.id].length === 2);
    function updateMethodColors() {
      const colors = darkMode.matches ? palette.dark : palette.light;
      const style = document.documentElement.style;
      for (const { id } of methods) {
        const [start, end] = colors[id];
        style.setProperty(`--m-${id}`, start);
        style.setProperty(`--m-${id}-end`, end || start);
        style.setProperty(`--m-${id}-paint`, end ? `linear-gradient(${start}, ${end})` : start);
      }
    }
    updateMethodColors();
    darkMode.addEventListener("change", updateMethodColors);
    const all = C.domains.flatMap((d) => d.benchmarks.map((b) => ({ d, b })));
    const wins = all.filter(({ b }) => b.scores[0] === Math.max(...b.scores)).length;
    const avgGap = all.reduce((a, { b }) => a + (b.scores[0] - b.scores[1]), 0) / all.length;
    const avgBase = all.reduce((a, { b }) => a + (b.scores[0] - Math.max(...b.scores.slice(2))), 0) / all.length;
    $("#perf-figures").innerHTML = [
      figure(`${wins}/${all.length}`, "", t("项基准中，自训 LoRA 得分最高")),
      figure(`+${avgGap.toFixed(1)}`, "pt", t("相对外部 Jev 的平均领先")),
      figure(`+${avgBase.toFixed(1)}`, "pt", t("相对最好的免训练 / Chat 基线")),
    ].join("");

    const root = $("#perf");
    let focus = null;
    methods.forEach((m) => {
      const b = h("button", { "aria-pressed": "false", title: `突出显示 ${m.label}` },
        `<span class="sw" style="background:${paintVar(m)}"></span>${m.label} <small>${esc(m.note)}</small>`);
      b.addEventListener("click", () => {
        focus = focus === m.id ? null : m.id;
        root.classList.toggle("focus", !!focus);
        $$(".legend button", root).forEach((x, i) => {
          x.classList.toggle("dim", !!focus && methods[i].id !== focus);
          x.setAttribute("aria-pressed", String(methods[i].id === focus));
        });
        $$(".bar, .bar-val", root).forEach((x) => x.classList.toggle("on", x.dataset.m === focus));
      });
      $("#perf-legend").appendChild(b);
    });

    const spans = { medical: 12, materials: 6, "grid-dist": 6, "grid-restore": 4, "grid-dispatch": 8, finance: 6, fusion: 6 };
    const chart = $("#perf-chart");
    const tip = $("#perf-tip");
    for (const d of C.domains) {
      const dom = h("div", { class: "domain", style: `--df-span:${spans[d.id] || 6}` });
      const head = h("div", { class: "domain-head" }, `<span class="no">${d.no}</span><h3>${esc(d.title)}</h3>`);
      if (d.trajectory) head.appendChild(h("button", { class: "replay-link", onclick: () => openTrajectory(d.trajectory, "evaluation") }, t("回放这次任务 →")));
      dom.appendChild(head);
      const row = h("div", { class: "bench-row", style: `--df-n:${d.benchmarks.length}` });
      for (const [i, b] of d.benchmarks.entries()) row.appendChild(benchCard(d, b, i));
      dom.appendChild(row);
      chart.appendChild(dom);
    }

    function benchCard(d, b, benchmarkIndex) {
      const W = 236, H = 148, top = 20, base = 124, left = 22;
      const plotH = base - top, bw = 22, gap = 12;
      const x0 = left + (W - left - (5 * bw + 4 * gap)) / 2;
      const best = Math.max(...b.scores);
      let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(b.name)}：${methods.map((m, i) => `${m.label} ${b.scores[i]}`).join("，")}">`;
      const gradientId = (m) => `perf-${d.id}-${benchmarkIndex}-${m.id}`;
      svg += `<defs>${gradients.map((m) => `<linearGradient id="${gradientId(m)}" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" style="stop-color:${colorVar(m)}"/><stop offset="1" style="stop-color:var(--m-${m.id}-end)"/>
      </linearGradient>`).join("")}</defs>`;
      [0, 50, 100].forEach((v) => {
        const y = base - (v / 100) * plotH;
        svg += `<line class="${v === 0 ? "base-line" : "grid-line"}" x1="${left}" x2="${W}" y1="${y}" y2="${y}"/>`;
        svg += `<text class="axis-lbl" x="${left - 4}" y="${y + 3}" text-anchor="end">${v}</text>`;
      });
      methods.forEach((m, i) => {
        const v = b.scores[i];
        const hgt = Math.max(1.5, (v / 100) * plotH);
        const x = x0 + i * (bw + gap), y = base - hgt, r = Math.min(3, hgt);
        const path = `M${x},${base} V${y + r} Q${x},${y} ${x + r},${y} H${x + bw - r} Q${x + bw},${y} ${x + bw},${y + r} V${base} Z`;
        const fill = gradients.includes(m) ? `url(#${gradientId(m)})` : colorVar(m);
        svg += `<path class="bar" data-m="${m.id}" d="${path}" fill="${fill}"/>`;
        svg += `<text class="bar-val ${v === best ? "best" : ""}" data-m="${m.id}" x="${x + bw / 2}" y="${y - 4}" text-anchor="middle">${v.toFixed(2)}</text>`;
        svg += `<text class="bar-key" x="${x + bw / 2}" y="${base + 13}" text-anchor="middle">${m.short}</text>`;
        svg += `<rect class="hit" data-i="${i}" x="${x - gap / 2}" y="${top - 14}" width="${bw + gap}" height="${base - top + 30}" fill="transparent"/>`;
      });
      svg += "</svg>";
      const card = h("div", { class: "bench" }, svg + `<div class="bench-name">${esc(b.name)}</div><div class="bench-desc">${esc(b.desc)}${b.note ? ` · ${esc(b.note)}` : ""}</div>`);
      if (b.secondary) {
        const sb = Math.max(...b.secondary.scores);
        card.appendChild(h("div", { class: "bench-sec" }, `${esc(b.secondary.label)} (%)<div class="vals">${b.secondary.scores.map((v) => `<span class="${v === sb ? "best" : ""}">${v.toFixed(2)}</span>`).join("")}</div>`));
      }
      $$(".hit", card).forEach((hit) => {
        const i = +hit.dataset.i;
        hit.addEventListener("mouseenter", () => {
          const m = methods[i];
          const rank = [...b.scores].sort((a, c) => c - a).indexOf(b.scores[i]) + 1;
          const extra = b.secondary ? `<div class="tt-row"><span>${esc(b.secondary.label)}</span><b>${b.secondary.scores[i].toFixed(2)}</b></div>` : "";
          const inv = b.invalid ? `<div class="tt-row"><span>${t("无效输出")}</span><b>${b.invalid[i]}</b></div>` : "";
          tip.innerHTML = `<div class="tt-title">${esc(b.name)}</div>
            <div class="tt-row"><i style="background:${paintVar(m)}"></i><span>${esc(m.label)}</span><b>${b.scores[i].toFixed(2)}%</b></div>
            ${extra}${inv}<div class="tt-sub">第 ${rank} / 5 · ${esc(d.title)}</div>`;
          tip.hidden = false;
          const rr = root.getBoundingClientRect(), hr = hit.getBoundingClientRect();
          let x = hr.left - rr.left + hr.width / 2 - tip.offsetWidth / 2;
          x = Math.max(0, Math.min(x, rr.width - tip.offsetWidth));
          tip.style.left = `${x}px`;
          tip.style.top = `${hr.top - rr.top - tip.offsetHeight - 6}px`;
        });
        hit.addEventListener("mouseleave", () => { tip.hidden = true; });
      });
      return card;
    }

    const rows = all.map(({ d, b }) => {
      const best = Math.max(...b.scores);
      return `<tr><td>${esc(d.title)}</td><td>${esc(b.name)}</td>${b.scores.map((v) => `<td class="${v === best ? "best" : ""}">${v.toFixed(2)}</td>`).join("")}</tr>`;
    }).join("");
    $("#perf-table").innerHTML = `<table><thead><tr><th>${t("领域")}</th><th>${t("基准")}</th>${methods.map((m) => `<th>${m.label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
    $$("[data-view]").forEach((btn) => btn.addEventListener("click", () => {
      $$("[data-view]").forEach((b) => b.classList.toggle("on", b === btn));
      $("#perf-chart").hidden = btn.dataset.view !== "chart";
      $("#perf-table").hidden = btn.dataset.view !== "table";
    }));
    $("#perf-notes").innerHTML = C.chartNotes.map((n) => `<li>${esc(n)}</li>`).join("");
  }

  // ------------------------------------------------------------------ DAG
  const DAG = { W: 440, colX: [96, 220, 344], top: 22, rowH: 50, nw: 148, nh: 34, dw: 140, dh: 38 };
  const nodePos = (id) => {
    const n = FLOW.nodes[id];
    return { x: n.x ?? DAG.colX[n.col], y: DAG.top + n.row * DAG.rowH, decision: n.kind === "decision" };
  };
  const nodeW = (id) => FLOW.nodes[id].w ?? (FLOW.nodes[id].kind === "decision" ? DAG.dw : DAG.nw);
  const halfW = (id) => nodeW(id) / 2;
  const halfH = (id) => (FLOW.nodes[id].kind === "decision" ? DAG.dh / 2 : DAG.nh / 2);

  function edgePath(a, b) {
    const A = nodePos(a), B = nodePos(b);
    if (a === "trainChoice" && b === "train") {
      const xm = DAG.colX[1];
      return { d: `M${A.x - halfW(a)},${A.y} H${xm} V${B.y} H${B.x + halfW(b)}`, lx: A.x - halfW(a) - 4, ly: A.y - 6, anchor: "end" };
    }
    if (A.y === B.y) {
      const dir = B.x > A.x ? 1 : -1;
      const sx = A.x + dir * halfW(a), ex = B.x - dir * halfW(b);
      return { d: `M${sx},${A.y} H${ex}`, lx: (sx + ex) / 2, ly: A.y - 6, anchor: "middle" };
    }
    if (B.y < A.y) {
      const dir = B.x > A.x ? 1 : -1;
      const sx = A.x + dir * halfW(a);
      return { d: `M${sx},${A.y} H${B.x} V${B.y + halfH(b)}`, lx: (sx + B.x) / 2, ly: A.y - 6, anchor: "middle" };
    }
    if (A.x === B.x) return { d: `M${A.x},${A.y + halfH(a)} V${B.y - halfH(b)}`, lx: A.x + 6, ly: (A.y + B.y) / 2 + 3, anchor: "start" };
    if (A.decision) {
      const dir = B.x > A.x ? 1 : -1;
      const sx = A.x + dir * halfW(a);
      return { d: `M${sx},${A.y} H${B.x} V${B.y - halfH(b)}`, lx: (sx + B.x) / 2, ly: A.y - 6, anchor: "middle" };
    }
    const my = B.y - halfH(b) - 12;
    return { d: `M${A.x},${A.y + halfH(a)} V${my} H${B.x} V${B.y - halfH(b)}`, lx: B.x, ly: my - 4, anchor: "middle" };
  }

  function loopPath(from, to) {
    const A = nodePos(from), B = nodePos(to);
    const x = 10, r = 8;
    const sx = A.x - halfW(from), ex = B.x - halfW(to);
    return `M${sx},${A.y} H${x + r} Q${x},${A.y} ${x},${A.y - r} V${B.y + r} Q${x},${B.y} ${x + r},${B.y} H${ex}`;
  }

  function renderDag(id) {
    const tc = C.trajectories[id];
    const rows = Math.max(...Object.values(FLOW.nodes).map((n) => n.row));
    const H = DAG.top + rows * DAG.rowH + 24;
    const pathSet = new Set(tc.path);
    const pathEdges = new Set(tc.path.slice(1).map((n, i) => `${tc.path[i]}>${n}`));
    let svg = `<svg viewBox="0 0 ${DAG.W} ${H}" role="img" aria-label="DecisionForge skills 流程图，按${esc(tc.tab)}任务实例化">
      <defs>
        <marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="#c4c3bd"/></marker>
        <marker id="ah-on" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="#111113"/></marker>
        <marker id="ah-loop" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" fill="#4F46E5"/></marker>
      </defs>`;
    for (const [a, b, label] of FLOW.edges) {
      const p = edgePath(a, b);
      const on = pathEdges.has(`${a}>${b}`);
      svg += `<path class="dag-edge ${on ? "path" : "off"}" data-edge="${a}>${b}" d="${p.d}" marker-end="url(#ah)"/>`;
      if (label) svg += `<text class="edge-lbl ${on ? "" : "off"}" data-elbl="${a}>${b}" x="${p.lx}" y="${p.ly}" text-anchor="${p.anchor}">${esc(label)}</text>`;
    }
    if (tc.loop) {
      const A = nodePos(tc.loop.from), B = nodePos(tc.loop.to);
      svg += `<path class="dag-edge loop" data-loop="1" d="${loopPath(tc.loop.from, tc.loop.to)}" marker-end="url(#ah-loop)"/>`;
      const ly = (A.y + B.y) / 2;
      svg += `<text class="edge-lbl loop" x="6" y="${ly}" transform="rotate(-90 6 ${ly})" text-anchor="middle" dy="-2">${esc(tc.loop.label)}</text>`;
    }
    for (const [nid, n] of Object.entries(FLOW.nodes)) {
      const { x, y } = nodePos(nid);
      const w = nodeW(nid);
      const cls = pathSet.has(nid) ? "path" : "off";
      let shape;
      if (n.kind === "decision") {
        const hw = w / 2, hh = DAG.dh / 2;
        const pts = `${x},${y - hh} ${x + hw},${y} ${x},${y + hh} ${x - hw},${y}`;
        shape = `<polygon class="shape" points="${pts}"/><polygon class="pulse" points="${pts}"/>`;
      } else {
        const rx = n.kind === "terminal" ? DAG.nh / 2 : 2;
        const rect = `x="${x - w / 2}" y="${y - DAG.nh / 2}" width="${w}" height="${DAG.nh}" rx="${rx}"`;
        shape = `<rect class="shape" ${rect}/><rect class="pulse" ${rect}/>`;
      }
      const showSkill = n.kind !== "decision";
      svg += `<g class="dag-node ${cls}" data-node="${nid}" tabindex="0" role="button" aria-label="${esc(n.label)}">
        <title>${esc(n.label)} · ${esc(n.skill)}${cls === "off" ? "（本任务未经过）" : ""}</title>
        ${shape}
        <text class="lbl" x="${x}" y="${showSkill ? y - 4 : y}">${esc(n.label)}</text>
        ${showSkill ? `<text class="sk" x="${x}" y="${y + 10}">${esc(n.skill)}</text>` : ""}
        <g class="round-badge" hidden><circle cx="${x + w / 2 - 2}" cy="${y - DAG.nh / 2 + 1}" r="8"/><text x="${x + w / 2 - 2}" y="${y - DAG.nh / 2 + 1}">×2</text></g>
      </g>`;
    }
    svg += "</svg>";
    const box = $("#dag");
    box.innerHTML = svg;
    $$(".dag-node", box).forEach((g) => {
      const nid = g.dataset.node;
      g.addEventListener("click", () => selectNode(nid, true));
      g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); selectNode(nid, true); } });
    });
  }

  /** Nodes with an event in [0, idx); decisions count once a later path node is reached; delivery once playback ends. */
  function visitedNodes(id, idx) {
    const tc = C.trajectories[id], events = T[id].events;
    const seen = new Set();
    for (let i = 0; i < idx; i++) seen.add(events[i].n);
    tc.path.forEach((nid, i) => {
      if (FLOW.nodes[nid].kind === "decision" && tc.path.slice(i + 1).some((x) => seen.has(x))) seen.add(nid);
    });
    if (idx >= events.length) seen.add(tc.path[tc.path.length - 1]);
    return seen;
  }

  function updateDag() {
    const id = state.traj, tc = C.trajectories[id];
    const events = T[id].events;
    const cur = state.idx > 0 ? events[state.idx - 1] : null;
    const order = tc.path;
    const seen = visitedNodes(id, state.idx);
    const roundReached = {};
    for (let i = 0; i < state.idx; i++) if (events[i].r > 1) roundReached[events[i].n] = true;
    const activeNode = state.ended ? null : cur && cur.n;
    $$(".dag-node").forEach((g) => {
      const nid = g.dataset.node;
      g.classList.toggle("visited", seen.has(nid));
      g.classList.toggle("active", nid === activeNode);
      g.classList.toggle("sel", nid === state.selected);
      const badge = $(".round-badge", g);
      if (badge) badge.toggleAttribute("hidden", !roundReached[nid]);
    });
    $$(".dag-edge[data-edge]").forEach((p) => {
      const [a, b] = p.dataset.edge.split(">");
      const ia = order.indexOf(a), ib = order.indexOf(b);
      const done = ia >= 0 && ib === ia + 1 && seen.has(a) && seen.has(b);
      p.classList.toggle("done", done);
      p.setAttribute("marker-end", done ? "url(#ah-on)" : "url(#ah)");
      const lbl = $(`[data-elbl="${a}>${b}"]`);
      if (lbl) lbl.classList.toggle("done", done);
    });
    const loop = $(".dag-edge[data-loop]");
    if (loop) loop.classList.toggle("done", Object.keys(roundReached).length > 0);
    if (!state.selected || state.followNode) renderNodeCard(state.ended ? order[order.length - 1] : cur ? cur.n : order[0]);
  }

  function selectNode(nid, jump) {
    state.selected = nid;
    state.followNode = false;
    const ns = DERIVED[state.traj].nodeStats[nid];
    if (jump && ns) seek((ns.first[1] ?? Object.values(ns.first)[0]) + 1, state.playing);
    renderNodeCard(nid);
    $$(".dag-node").forEach((g) => g.classList.toggle("sel", g.dataset.node === nid));
  }

  function renderNodeCard(nid) {
    const id = state.traj, tc = C.trajectories[id];
    const n = FLOW.nodes[nid];
    if (!n) return;
    const ns = DERIVED[id].nodeStats[nid];
    const inPath = tc.path.includes(nid);
    const events = T[id].events;
    const curEv = state.idx > 0 ? events[state.idx - 1] : null;
    const active = curEv && curEv.n === nid && !state.ended;
    const done = inPath && visitedNodes(id, state.idx).has(nid);
    const stateLbl = !inPath ? `<span class="state">本任务未经过</span>`
      : active ? `<span class="state active">进行中${curEv.r > 1 ? ` · 第 ${curEv.r} 轮` : ""}</span>`
      : done ? `<span class="state">已完成</span>` : `<span class="state">尚未到达</span>`;
    const lines = tc.nodes[nid] || (inPath ? [] : [t("这一分支用于其他情况，例如只有场景描述、没有自有数据，或选择不训练。")]);
    let stats = "";
    if (ns) {
      const rounds = [...ns.rounds].sort();
      stats = `<div class="nstats">${ns.msgs} 条消息 · ${fmtNum(ns.tools)} 次命令${ns.asks ? ` · ${ns.asks} 次确认` : ""}${rounds.length > 1 ? ` · ${rounds.length} 轮` : ""}</div>`;
    }
    const jumps = ns ? `<div class="jumps">${Object.keys(ns.first).sort().map((r) => `<button data-jump="${ns.first[r]}">${Object.keys(ns.first).length > 1 ? `跳到第 ${r} 轮` : "跳到这一阶段"} →</button>`).join("")}</div>` : "";
    const card = $("#node-card");
    card.innerHTML = `<h4>${esc(n.label)} ${stateLbl}</h4><p class="skill">${esc(n.skill)} · ${esc(C.skills[n.skill] || "")}</p>
      ${lines.length ? `<ul>${lines.map((l) => `<li>${inline(l)}</li>`).join("")}</ul>` : ""}${stats}${jumps}`;
    $$("[data-jump]", card).forEach((b) => b.addEventListener("click", () => { seek(+b.dataset.jump + 1, state.playing); renderNodeCard(nid); }));
  }

  // ------------------------------------------------------------------ player
  const SPEEDS = [1, 2, 4, 8, 16, 32];
  const state = {
    traj: TRAJ_IDS[0], idx: 0, playing: false, speed: 4, timer: null, anim: null,
    selected: null, followNode: true, ended: false, pos: {}, follow: true,
  };
  const tr = () => $("#transcript");
  const stageTag = (ev) => `<span class="stage-tag">${esc(nodeLabel(ev.n))}${ev.r > 1 ? ` · 第 ${ev.r} 轮` : ""}</span>`;

  function makeEvent(ev, i, animate) {
    const harness = T[state.traj].harness;
    let el, dur = 0, finish = () => {}, tick = null;
    const k = ev.k;
    if (k === "session") {
      el = h("div", { class: "ev ev-session" }, `<span><b>${esc(ev.x)}</b>${esc(harness)} · ${fmtTime(ev.t)}</span>`);
      dur = 900;
    } else if (k === "user") {
      el = h("div", { class: "ev ev-user" }, `<div class="ev-meta"><span class="who">你</span><time>${fmtTime(ev.t)}</time></div><div class="bubble">${inline(ev.x).replace(/\n/g, "<br>")}</div>`);
      dur = Math.min(2600, 900 + ev.x.length * 6);
    } else if (k === "assistant") {
      el = h("div", { class: "ev ev-assistant" }, `<div class="ev-meta"><span class="who">${esc(harness)}</span><time>${fmtTime(ev.t)}</time>${stageTag(ev)}</div><div class="body md"></div>`);
      const body = $(".body", el);
      if (animate) {
        const text = ev.x;
        const typeMs = Math.min(5200, 500 + text.length * 14);
        dur = typeMs + 500;
        const typer = h("div", { class: "typing" });
        body.appendChild(typer);
        let shown = 0;
        tick = (elapsed) => {
          const n = Math.min(text.length, Math.ceil((elapsed / typeMs) * text.length));
          if (n !== shown) { typer.textContent = text.slice(0, n); shown = n; }
          if (n >= text.length) { body.innerHTML = md(text); tick = null; return true; }
          return false;
        };
        finish = () => { body.innerHTML = md(text); };
      } else {
        body.innerHTML = md(ev.x);
      }
    } else if (k === "status") {
      el = h("div", { class: "ev ev-status" }, `<time>${fmtClock(ev.t)}</time><span>${inline(ev.x)}</span>`);
      dur = 450;
    } else if (k === "subagent") {
      el = h("div", { class: "ev ev-subagent" }, `<details class="sub-card"><summary>派出独立审阅子代理 <b>${esc(ev.name)}</b>${ev.model ? ` · ${esc(ev.model)}` : ""} · ${fmtClock(ev.t)}</summary><div class="md">${md(ev.x || t("审阅结论已回传主会话。"))}</div></details>`);
      dur = 1200;
    } else if (k === "ask") {
      el = h("div", { class: "ev ev-ask" });
      const card = h("div", { class: `ask-card ${animate ? "ask-pending" : ""}` });
      let html = `<div class="ask-label">需要你确认 · ${fmtClock(ev.t)} · ${esc(nodeLabel(ev.n))}</div>`;
      ev.questions.forEach((q, qi) => {
        const ans = ev.answers.find((a) => a.q.slice(0, 30) === q.q.slice(0, 30)) || ev.answers[qi];
        const chosen = ans ? ans.a : null;
        const hit = q.options.some((o) => o === chosen);
        html += `<div class="ask-q">${q.header ? `<span class="hdr">${esc(q.header)}</span>` : ""}${inline(q.q)}</div>`;
        if (q.options.length) html += `<div class="ask-opts">${q.options.map((o, oi) => `<span class="ask-opt ${o === chosen ? "chosen" : ""}" title="${esc((q.details || [])[oi] || "")}">${esc(o)}</span>`).join("")}</div>`;
        if (chosen && !hit) html += `<div class="ask-free"><b>你的回答</b>${esc(chosen)}</div>`;
      });
      card.innerHTML = html;
      el.appendChild(card);
      dur = 2600;
      if (animate) {
        tick = (elapsed) => { if (elapsed > 1300) { card.classList.remove("ask-pending"); tick = null; return true; } return false; };
        finish = () => card.classList.remove("ask-pending");
      }
    } else if (k === "tools") {
      el = h("div", { class: "ev ev-tools" });
      const counts = {};
      let errs = 0;
      ev.items.forEach((it) => { counts[it.n] = (counts[it.n] || 0) + 1; errs += it.e ? 1 : 0; });
      const names = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([n, c]) => `${n}${c > 1 ? `×${c}` : ""}`).join(" ");
      const t0 = ev.items[0].t, t1 = ev.items[ev.items.length - 1].t;
      const span = fmtClock(t0) + (fmtClock(t1) !== fmtClock(t0) ? `–${fmtClock(t1)}` : "");
      const card = h("div", { class: "tools-card" });
      const head = h("button", { class: "tools-head", "aria-expanded": "false" },
        `<span class="caret">▸</span><span>${ev.items.length} 条命令</span><span class="names">${esc(names)}</span>${errs ? `<span class="err">${errs} 个报错</span>` : ""}<span>${span}</span>`);
      card.appendChild(head);
      head.addEventListener("click", () => {
        const open = !card.classList.contains("open");
        card.classList.toggle("open", open);
        head.setAttribute("aria-expanded", String(open));
        let list = $(".tools-list", card);
        if (open && !list) { list = toolList(ev.items); card.appendChild(list); }
        if (list) list.hidden = !open;
        const live = $(".tools-live", card);
        if (live) live.remove();
      });
      el.appendChild(card);
      if (animate) {
        dur = Math.min(3200, 300 + ev.items.length * 120);
        const live = h("div", { class: "tools-live" });
        card.appendChild(live);
        let shown = 0;
        tick = (elapsed) => {
          const n = Math.min(ev.items.length, Math.floor(((elapsed - 150) / Math.max(1, dur - 400)) * ev.items.length) + 1);
          while (shown < n && shown < ev.items.length) {
            const it = ev.items[shown++];
            live.appendChild(h("div", { class: it.e ? "e" : "" }, esc(it.c)));
            while (live.children.length > 6) live.firstChild.remove();
          }
          if (shown >= ev.items.length && elapsed >= dur - 150) { tick = null; return true; }
          return false;
        };
        finish = () => { const l = $(".tools-live", card); if (l) l.remove(); };
      }
    }
    el.dataset.idx = i;
    return { el, dur, finish: () => finish(), tick: (e) => (tick ? tick(e) : true) };
  }

  function toolList(items) {
    const list = h("div", { class: "tools-list" });
    items.forEach((it) => {
      const row = h("div", { class: `titem ${it.e ? "e" : ""}` });
      const btn = h("button", { class: "titem-head" }, `<span class="c">${esc(it.c)}</span><time>${fmtClock(it.t)}</time>`);
      btn.addEventListener("click", () => {
        let pre = $("pre", row);
        if (pre) { pre.remove(); return; }
        pre = h("pre", {}, `${esc(it.b || it.c)}${it.o ? `\n\n<span class="out">${esc(it.o)}</span>` : ""}`);
        row.appendChild(pre);
      });
      row.appendChild(btn);
      list.appendChild(row);
    });
    return list;
  }

  function gapBefore(i) {
    const events = T[state.traj].events;
    if (i === 0 || events[i].k === "session") return null;
    const d = new Date(events[i].t) - new Date(events[i - 1].t);
    if (d < 40 * 60000) return null;
    return h("div", { class: "gap" }, `${fmtDur(d)}后 · ${fmtTime(events[i].t)}`);
  }

  function isHidden(ev) {
    const t = tr();
    if (ev.k === "tools") return t.classList.contains("hide-tools");
    if (ev.k === "status") return t.classList.contains("hide-status");
    return false;
  }

  function appendEvent(i, animate) {
    const g = gapBefore(i);
    if (g) tr().appendChild(g);
    const ev = T[state.traj].events[i];
    const made = makeEvent(ev, i, animate && !isHidden(ev));
    tr().appendChild(made.el);
    return made;
  }

  const nearBottom = () => { const t = tr(); return t.scrollHeight - t.scrollTop - t.clientHeight < 140; };
  const scrollDown = () => { if (state.follow) tr().scrollTop = tr().scrollHeight; };

  function stopAnim() {
    if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    if (state.anim) { cancelAnimationFrame(state.anim.raf); state.anim.made.finish(); state.anim = null; }
  }

  function playStep() {
    const events = T[state.traj].events;
    if (state.idx >= events.length) { finishPlayback(); return; }
    const i = state.idx;
    const made = appendEvent(i, true);
    state.idx = i + 1;
    afterAdvance();
    scrollDown();
    const hidden = isHidden(events[i]);
    const total = hidden ? 0 : made.dur / state.speed;
    const start = performance.now();
    const anim = { made, raf: 0 };
    state.anim = anim;
    const frame = (now) => {
      const elapsed = (now - start) * state.speed;
      const done = hidden ? true : made.tick(elapsed);
      if (!hidden) scrollDown();
      if (done && now - start >= total) {
        made.finish();
        state.anim = null;
        if (state.playing) state.timer = setTimeout(playStep, hidden ? 0 : Math.max(40, 220 / state.speed));
        return;
      }
      anim.raf = requestAnimationFrame(frame);
    };
    anim.raf = requestAnimationFrame(frame);
  }

  function play() {
    if (state.playing) return;
    if (state.idx >= T[state.traj].events.length) seek(0);
    $(".intro-card", tr())?.remove();
    state.playing = true;
    state.ended = false;
    state.followNode = true;
    state.selected = null;
    updatePlayBtn();
    playStep();
  }
  function pause() {
    state.playing = false;
    if (state.timer) { clearTimeout(state.timer); state.timer = null; }
    updatePlayBtn();
  }
  function finishPlayback() {
    state.playing = false;
    state.ended = true;
    updatePlayBtn();
    afterAdvance();
  }
  function updatePlayBtn() {
    const b = $("#btn-play");
    b.textContent = state.playing ? t("暂停") : t("播放");
    b.setAttribute("aria-label", b.textContent);
  }

  /** Render events [0, target) instantly; continue playback afterwards if keepPlaying. */
  function seek(target, keepPlaying = state.playing) {
    const events = T[state.traj].events;
    target = Math.max(0, Math.min(events.length, target));
    stopAnim();
    state.playing = false;
    const t = tr();
    t.classList.add("no-anim");
    t.innerHTML = "";
    const frag = document.createDocumentFragment();
    for (let i = 0; i < target; i++) {
      const g = gapBefore(i);
      if (g) frag.appendChild(g);
      frag.appendChild(makeEvent(events[i], i, false).el);
    }
    t.appendChild(frag);
    state.idx = target;
    state.ended = target >= events.length;
    if (target === 0) showIntro();
    requestAnimationFrame(() => {
      t.classList.remove("no-anim");
      state.follow = true;
      $("#btn-follow").hidden = true;
      if (target > 0) {
        const anchor = t.querySelector(`[data-idx="${target - 1}"]`);
        t.scrollTop = anchor ? Math.max(0, anchor.offsetTop - 40) : t.scrollHeight;
        state.follow = keepPlaying || nearBottom();
        $("#btn-follow").hidden = state.follow;
      }
    });
    afterAdvance();
    if (keepPlaying && target < events.length) {
      state.playing = true;
      state.ended = false;
      state.timer = setTimeout(playStep, 120);
    }
    updatePlayBtn();
  }

  function showIntro() {
    const id = state.traj, tc = C.trajectories[id], s = T[id].stats;
    const card = h("div", { class: "intro-card" }, `
      <p class="kicker">${esc(T[id].harness)} · ${esc(tc.model)}</p>
      <h4>${esc(tc.title)}</h4>
      <p>${s.sessions} 个会话 · ${s.user} 条需求 · ${s.asks} 次确认 · ${fmtNum(s.tools)} 条命令</p>
      <p>跨越 ${fmtDur(new Date(T[id].end) - new Date(T[id].start))}，其中 agent 活跃约 ${s.activeHours} 小时。</p>`);
    card.appendChild(h("button", { class: "btn primary", onclick: () => play() }, t("开始回放")));
    tr().appendChild(card);
  }

  function afterAdvance() {
    const id = state.traj, events = T[id].events, d = DERIVED[id];
    $$("#scrub-track span").forEach((s) => {
      const st = +s.dataset.start, en = +s.dataset.end;
      const f = Math.max(0, Math.min(1, (state.idx - st) / (en - st)));
      s.firstChild.style.width = `${f * 100}%`;
    });
    const cur = state.idx > 0 ? events[state.idx - 1] : null;
    const c = cur ? d.cum[state.idx - 1] : { msgs: 0, tools: 0 };
    let sessionNo = 1;
    for (let i = state.idx - 1; i >= 0; i--) if (events[i].k === "session") { sessionNo = events[i].s; break; }
    const elapsed = cur ? new Date(cur.t) - new Date(T[id].start) : 0;
    $("#clock").innerHTML = `
      <span>时间 <b>${fmtTime(cur ? cur.t : T[id].start)}</b></span>
      <span>会话 <b>${sessionNo}/${T[id].stats.sessions}</b></span>
      <span>已用时 <b>${fmtDur(elapsed)}</b></span>
      <span>阶段 <b>${cur ? esc(nodeLabel(cur.n)) + (cur.r > 1 ? ` · 第 ${cur.r} 轮` : "") : "—"}</b></span>
      <span>消息 <b>${c.msgs}/${d.totalMsgs}</b></span>
      <span>命令 <b>${fmtNum(c.tools)}/${fmtNum(d.totalTools)}</b></span>`;
    updateDag();
  }

  function renderScrubber() {
    const id = state.traj, d = DERIVED[id], n = T[id].events.length;
    const track = $("#scrub-track");
    track.innerHTML = "";
    for (const r of d.runs) {
      track.appendChild(h("span", { "data-start": r.start, "data-end": r.end, style: `flex:${r.end - r.start} 0 0` }, "<b></b>"));
    }
    const sc = $("#scrubber"), tip = $("#scrub-tip");
    const frac = (e) => { const rect = sc.getBoundingClientRect(); return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)); };
    sc.onmousemove = (e) => {
      const f = frac(e);
      const ev = T[id].events[Math.min(n - 1, Math.floor(f * n))];
      tip.hidden = false;
      tip.style.left = `${f * 100}%`;
      tip.textContent = `${nodeLabel(ev.n)}${ev.r > 1 ? ` · 第 ${ev.r} 轮` : ""} · ${fmtTime(ev.t)}`;
    };
    sc.onmouseleave = () => { tip.hidden = true; };
    sc.onclick = (e) => { state.followNode = true; state.selected = null; seek(Math.round(frac(e) * n)); };
  }

  function renderTrajHead() {
    const id = state.traj, tc = C.trajectories[id], s = T[id].stats;
    const results = tc.results.map((r) => {
      const dlt = r.ours - r.jev;
      return `<div class="res"><div class="lbl">${esc(r.label)}</div>
        <div class="ours">${r.ours.toFixed(2)}<small>%</small></div>
        <div class="bars"><i style="width:${r.ours}%;background:var(--m-lora-paint)"></i><i style="width:${r.jev}%;background:var(--m-jev-paint)"></i></div>
        <div class="vs">Jev ${r.jev.toFixed(2)} · <span class="delta ${dlt >= 0 ? "up" : "down"}">${dlt >= 0 ? "+" : ""}${dlt.toFixed(2)}</span></div></div>`;
    }).join("");
    $("#traj-head").innerHTML = `
      <div class="traj-intro"><h3>${esc(tc.title)}</h3><p>${esc(tc.pitch)}</p>
        <div class="meta">${esc(tc.harness)} · ${esc(tc.model)} · ${s.sessions} 个会话 · ${fmtNum(s.tools)} 条命令 · 活跃约 ${s.activeHours} 小时</div></div>
      <div class="traj-results">${results}</div>`;
  }

  function switchTraj(id, { reset = false } = {}) {
    if (!T[id]) return;
    if (state.traj !== id || reset) {
      stopAnim();
      state.pos[state.traj] = state.idx;
      state.playing = false;
      state.traj = id;
      state.selected = null;
      state.followNode = true;
    }
    $$(".tab").forEach((b) => { b.classList.toggle("on", b.dataset.id === id); b.setAttribute("aria-selected", String(b.dataset.id === id)); });
    renderTrajHead();
    renderDag(id);
    renderScrubber();
    seek(state.pos[id] || 0, false);
  }

  function openTrajectory(id, node, round, index) {
    if (!T[id]) return;
    if (state.traj !== id) switchTraj(id);
    const stage = $("#stage");
    window.scrollTo({ top: stage.getBoundingClientRect().top + window.scrollY - 72, behavior: "smooth" });
    if (index != null) { state.followNode = true; state.selected = null; seek(index + 1, false); return; }
    if (node) {
      const ns = DERIVED[id].nodeStats[node];
      if (ns) {
        state.selected = null;
        state.followNode = true;
        seek((ns.first[round || 1] ?? Object.values(ns.first)[0]) + 1, false);
        return;
      }
    }
    if (state.idx === 0) play();
  }

  function nextStage() {
    const id = state.traj, runs = DERIVED[id].runs;
    const next = runs.find((r) => r.start > state.idx - 1);
    state.selected = null;
    state.followNode = true;
    seek(next ? next.start + 1 : T[id].events.length);
  }

  function initPlayer() {
    $("#replay-title").innerHTML = C.replay.title;
    $("#replay-body").textContent = C.replay.body;
    const tabs = $("#traj-tabs");
    for (const id of TRAJ_IDS) {
      const tc = C.trajectories[id];
      tabs.appendChild(h("button", { class: "tab", role: "tab", "data-id": id, onclick: () => switchTraj(id) },
        `<span class="t">${esc(tc.tab)}</span><span class="h">${esc(tc.harness)} · ${esc(tc.model)}</span>`));
    }
    const speed = $("#speed");
    SPEEDS.forEach((s) => {
      const b = h("button", { class: s === state.speed ? "on" : "" }, `${s}×`);
      b.addEventListener("click", () => {
        state.speed = s;
        $$("button", speed).forEach((x) => x.classList.toggle("on", x === b));
      });
      speed.appendChild(b);
    });
    $("#btn-play").addEventListener("click", () => (state.playing ? pause() : play()));
    $("#btn-next").addEventListener("click", () => {
      const events = T[state.traj].events;
      let j = state.idx;
      while (j < events.length && !["user", "assistant", "ask"].includes(events[j].k)) j++;
      seek(Math.min(events.length, j + 1));
    });
    $("#btn-stage").addEventListener("click", nextStage);
    $("#btn-end").addEventListener("click", () => { pause(); seek(T[state.traj].events.length, false); });
    $("#btn-reset").addEventListener("click", () => { pause(); state.selected = null; state.followNode = true; seek(0, false); });
    $("#f-tools").addEventListener("change", (e) => tr().classList.toggle("hide-tools", !e.target.checked));
    $("#f-status").addEventListener("change", (e) => tr().classList.toggle("hide-status", !e.target.checked));
    tr().addEventListener("scroll", () => {
      state.follow = nearBottom();
      $("#btn-follow").hidden = state.follow;
    });
    $("#btn-follow").addEventListener("click", () => { state.follow = true; tr().scrollTop = tr().scrollHeight; $("#btn-follow").hidden = true; });

    document.addEventListener("keydown", (e) => {
      if (/input|textarea|select/i.test(document.activeElement?.tagName || "")) return;
      const sec = $("#stage").getBoundingClientRect();
      if (sec.bottom < 0 || sec.top > window.innerHeight) return;
      if (e.code === "Space") { e.preventDefault(); state.playing ? pause() : play(); }
      else if (e.key === "ArrowRight" && e.shiftKey) { e.preventDefault(); nextStage(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); $("#btn-next").click(); }
    });

    $$("[data-open]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); openTrajectory(a.dataset.open); }));
    switchTraj(state.traj, { reset: true });
  }

  function applyHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    const id = p.get("traj");
    if (id && T[id]) openTrajectory(id, p.get("node"), +(p.get("round") || 1));
  }

  // ------------------------------------------------------------------ boot
  initTopbar();
  renderHero();
  renderIntro();
  renderSteps();
  renderGates();
  renderPerformance();
  initPlayer();
  applyHash();
  window.addEventListener("hashchange", applyHash);
})();
