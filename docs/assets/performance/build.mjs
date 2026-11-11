// Build the README performance overview from the demo's benchmark data.
// Usage: node docs/assets/performance/build.mjs
// Output: docs/assets/performance/overview-{zh,en}-{light,dark}.svg
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..', '..');

// content.js assigns window.DFORGE_COPY = { zh, en }; evaluate it in a sandbox.
const sandbox = { window: {}, structuredClone };
vm.runInNewContext(readFileSync(join(root, 'demo/assets/content.js'), 'utf8'), sandbox);
const copy = sandbox.window.DFORGE_COPY;

const COPY = {
  zh: {
    title: '7 个领域、15 项基准上的平均准确率',
    subtitle: '每个领域取其各项基准 Accuracy 的平均值；逐项结果、表格与 agent 回放见 Demo。',
    methods: ['DecisionForge 自训（LoRA）', '外部 Jev', 'LLM2Jev（免训练）', 'AnyJev（免训练）', 'Chat（未微调）'],
    domains: ['医疗分诊', '材料性质', '配电网控制', '停电恢复', '调度与储能', '金融文本', '聚变诊断'],
  },
  en: {
    title: 'Mean accuracy across 7 domains and 15 benchmarks',
    subtitle: 'Each domain averages the Accuracy of its benchmarks. See the demo for per-benchmark results, tables and agent replays.',
    methods: ['DecisionForge (LoRA)', 'External Jev', 'LLM2Jev (training-free)', 'AnyJev (training-free)', 'Chat (untuned)'],
    domains: ['Medical triage', 'Materials', 'Grid control', 'Restoration', 'Dispatch & storage', 'Finance text', 'Fusion plasma'],
  },
};

// Categorical palette in demo method order; dark steps validated separately
// (dataviz validate_palette.js, surface #0d1117).
const THEME = {
  light: {
    bg: '#ffffff', ink: '#1f2328', muted: '#59636e', grid: '#e6e8eb', axis: '#c8ced6',
    series: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'],
  },
  dark: {
    bg: '#0d1117', ink: '#f0f6fc', muted: '#9198a1', grid: '#21262d', axis: '#3d444d',
    series: ['#3a84e0', '#e0602c', '#1aa874', '#c08600', '#d06892'],
  },
};

const FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif";
const W = 1000, H = 460;
const plot = { left: 56, right: W - 24, top: 132, bottom: H - 64 };
const BAR = 16, GAP = 2;

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const domains = copy.zh.domains.map((d) => ({
  benchmarks: d.benchmarks.length,
  means: [0, 1, 2, 3, 4].map((m) => mean(d.benchmarks.map((b) => b.scores[m]))),
}));
const benchmarkCount = domains.reduce((n, d) => n + d.benchmarks, 0);
if (domains.length !== 7 || benchmarkCount !== 15) {
  throw new Error(`Copy says 7 domains / 15 benchmarks, data has ${domains.length} / ${benchmarkCount}`);
}

// Approximate rendered width: CJK and full-width glyphs ~1em, Latin ~0.56em.
const textWidth = (s, size) => [...s].reduce((w, ch) => w + (/[\u3000-\u9fff\uff00-\uffef]/.test(ch) ? size : size * 0.56), 0);
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const y = (v) => plot.bottom - (v / 100) * (plot.bottom - plot.top);

// Bar with a 4px rounded data-end, flat on the baseline.
function bar(x, value, color) {
  const top = y(value), r = Math.min(4, (plot.bottom - top) / 2, BAR / 2);
  return `<path d="M${x} ${plot.bottom}V${(top + r).toFixed(1)}Q${x} ${top.toFixed(1)} ${x + r} ${top.toFixed(1)}H${x + BAR - r}Q${x + BAR} ${top.toFixed(1)} ${x + BAR} ${(top + r).toFixed(1)}V${plot.bottom}Z" fill="${color}"/>`;
}

function render(lang, mode) {
  const t = COPY[lang], c = THEME[mode];
  const out = [];
  out.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="t d">`);
  out.push(`<title id="t">${esc(t.title)}</title>`);
  const desc = domains.map((d, i) => `${t.domains[i]}: ${d.means.map((v, m) => `${t.methods[m]} ${v.toFixed(1)}%`).join(', ')}`).join('; ');
  out.push(`<desc id="d">${esc(desc)}</desc>`);
  out.push(`<rect width="${W}" height="${H}" rx="12" fill="${c.bg}"/>`);
  out.push(`<g font-family="${FONT}">`);
  out.push(`<text x="24" y="38" font-size="20" font-weight="600" fill="${c.ink}">${esc(t.title)}</text>`);
  out.push(`<text x="24" y="62" font-size="13" fill="${c.muted}">${esc(t.subtitle)}</text>`);

  // Legend: one row, swatch + label in text ink.
  let lx = 24;
  t.methods.forEach((label, m) => {
    out.push(`<rect x="${lx}" y="86" width="12" height="12" rx="3" fill="${c.series[m]}"/>`);
    out.push(`<text x="${lx + 18}" y="96.5" font-size="13" fill="${c.ink}"${m === 0 ? ' font-weight="600"' : ''}>${esc(label)}</text>`);
    lx += 18 + textWidth(label, 13) + 24;
  });

  // Recessive grid and y labels.
  for (const v of [0, 25, 50, 75, 100]) {
    const gy = y(v).toFixed(1);
    out.push(`<line x1="${plot.left}" x2="${plot.right}" y1="${gy}" y2="${gy}" stroke="${v === 0 ? c.axis : c.grid}" stroke-width="1"/>`);
    out.push(`<text x="${plot.left - 10}" y="${(+gy + 4).toFixed(1)}" font-size="12" text-anchor="end" fill="${c.muted}">${v}%</text>`);
  }

  const groupW = (plot.right - plot.left) / domains.length;
  const clusterW = 5 * BAR + 4 * GAP;
  domains.forEach((d, i) => {
    const x0 = plot.left + i * groupW + (groupW - clusterW) / 2;
    d.means.forEach((v, m) => {
      out.push(bar(x0 + m * (BAR + GAP), v, c.series[m]));
    });
    // Direct label only on the DecisionForge bar.
    const lead = d.means[0];
    out.push(`<text x="${(x0 + BAR / 2).toFixed(1)}" y="${(y(lead) - 7).toFixed(1)}" font-size="12" font-weight="600" text-anchor="middle" fill="${c.ink}">${lead.toFixed(1)}</text>`);
    const cx = (plot.left + i * groupW + groupW / 2).toFixed(1);
    out.push(`<text x="${cx}" y="${plot.bottom + 24}" font-size="13" text-anchor="middle" fill="${c.ink}">${esc(t.domains[i])}</text>`);
    const n = lang === 'zh' ? `${d.benchmarks} 项基准` : `${d.benchmarks} benchmark${d.benchmarks > 1 ? 's' : ''}`;
    out.push(`<text x="${cx}" y="${plot.bottom + 42}" font-size="11" text-anchor="middle" fill="${c.muted}">${n}</text>`);
  });

  out.push('</g></svg>');
  return out.join('\n') + '\n';
}

for (const lang of ['zh', 'en']) {
  for (const mode of ['light', 'dark']) {
    const file = join(here, `overview-${lang}-${mode}.svg`);
    writeFileSync(file, render(lang, mode));
    console.log(file);
  }
}
