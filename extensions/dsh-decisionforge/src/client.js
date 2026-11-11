import React from 'react';
import logo from '../assets/icon.png';
import styles from './style.css';
import { content } from './content.js';
import translations from './translations.json';
const Language = React.createContext('zh');
function useCopy() {
  const language = React.useContext(Language);
  return text => language === 'en' ? translations[text] : text;
}

import { selectLaunchWorkspace } from './launch-workspace.js';

const h = React.createElement;
export const inject = ['slots', 'layout', 'uiWorkspace', 'workspaces', 'sessions', 'locale'];
const PANEL = 'dforge-guide';

function BrandMark({ size = 32, className }) {
  return h('img', { src: logo, alt: 'DecisionForge', width: size, height: size, className, style: { objectFit: 'contain', borderRadius: '22%' } });
}
function BrandName() {
  return h('span', { className: 'df-brand-name' }, 'DecisionForge');
}
function GuideIcon({ size = 18 }) {
  return h('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.6, 'aria-hidden': true },
    h('path', { d: 'M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1m0-15c3-2 6-2 9-1v15c-3-1-6-1-9 1M12 5v15' }));
}
function CopyExample({ text, label }) {
  const t = useCopy();
  label ??= t('复制示例');
  const [status, setStatus] = React.useState('');
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setStatus(t("已复制，返回对话后粘贴并修改"));
    } catch {
      setStatus(t("未能访问剪贴板，请选中下方文本手动复制"));
    }
  }
  return h('div', { className: 'df-example' },
    h('pre', { tabIndex: 0 }, text),
    h('div', { className: 'df-copy-row' },
      h('button', { type: 'button', className: 'df-button df-button-small', onClick: copy }, label),
      h('span', { role: 'status', 'aria-live': 'polite' }, status)));
}
function Section({ id, number, title, intro, children }) {
  return h('section', { id, className: 'df-section', 'aria-labelledby': id + '-title' },
    h('div', { className: 'df-section-heading' }, h('span', { className: 'df-number' }, number), h('h2', { id: id + '-title' }, title)),
    intro && h('p', { className: 'df-muted' }, intro), children);
}
function Guide({ onChat }) {
  const language = React.useContext(Language);
  const t = useCopy();
  const { skills, workflow, resumeExample, inspectExample } = content[language];
  return h('main', { className: 'df-guide' },
    h('div', { className: 'df-guide-inner' },
      h('header', { className: 'df-topbar' },
        h('div', { className: 'df-topbar-brand' }, h(BrandMark, { size: 28 }), h('span', null, 'DecisionForge'), h('span', { className: 'df-divider' }, '/'), h('span', { className: 'df-muted' }, t("使用指南"))),
        h('button', { type: 'button', className: 'df-button', onClick: onChat }, t("返回对话"), h('span', { 'aria-hidden': true }, ' ↗'))),
      h('div', { className: 'df-intro' },
        h('span', { className: 'df-eyebrow' }, t("从场景到决策模型")),
        h('h1', null, t("把目标告诉 Agent，"), h('br'), t("一步步完成 DecisionForge 任务。")),
        h('p', null, t("数据准备、模型训练、评测与分析，通过五个项目 skills 协作完成。你决定目标和资源，Agent 展示方案、执行命令，并保留可追溯的结果。")),
        h('nav', { className: 'df-jumplinks', 'aria-label': t("指南目录") },
          ...[['start', t("开始使用")], ['workflow', t("了解流程")], ['skills', t("选择 skill")], ['results', t("结果与接续")]].map(([id, label]) => h('a', { href: '#' + id, key: id }, label)))),
      h(Section, { id: 'start', number: '01', title: t("开始之前，先连接好项目") },
        h('ol', { className: 'df-setup' },
          ...[
            [t("配置 Agent 模型"), t("在设置 → 模型中配置可用服务，并在聊天框中选择模型。Agent 聊天凭据与生成、Checker、检索、MinerU、评测等业务凭据分别配置。")],
            [t("选择工作区"), t("通过项目的 npm start 启动时，会自动添加并选择当前 checkout。直接运行 dsh web 时，请手动添加并选择完整 DecisionForge 仓库；之后可随时切换工作区。")],
            [t("核查 Python 环境"), t("在激活所需 Python 环境的终端启动界面。第一条消息先核查 pwd、sys.executable 和 README，确保命令实际运行在正确目录和解释器。")],
          ].map(([title, detail]) => h('li', { key: title }, h('strong', null, title), h('p', null, detail)))),
        h('details', { className: 'df-details' }, h('summary', null, t("复制一条只读预检请求")), h(CopyExample, { text: inspectExample }))),
      h(Section, { id: 'workflow', number: '02', title: t("每一步都有明确的方案与结果"), intro: t("以下是端到端路线。只需要某个阶段时，直接调用对应 skill；已有合格数据或报告也可以复用。") },
        h('ol', { className: 'df-flow', 'aria-label': t("DecisionForge 阶段流程") }, ...workflow.map(([title, detail], i) => h('li', { key: title }, h('span', { className: 'df-flow-index' }, String(i + 1).padStart(2, '0')), h('strong', null, title), h('span', null, detail)))),
        h('div', { className: 'df-note' }, h('strong', null, t("你可以接受、修改或暂停。")), h('p', null, t("总体路线确认只确定范围和顺序。每个具体业务阶段先展示策略、配置、资源与产物，得到接受后才执行；未回复不等于接受，仍有效的已有授权继续沿用。"))),
        h('p', { className: 'df-muted' }, t("有自有数据：先固定独立评测集，再准备训练数据。只有场景：先寻找资料、整理数据，再选择训练或使用已有模型。训练、验证与最终测试按来源隔离。"))),
      h(Section, { id: 'skills', number: '03', title: t("从 /dforge 开始，或直接选一个阶段"), intro: t("在聊天中输入 /名称 加任务描述。先提供已有的文件路径、配置、模型和资源；缺少的决定由 Agent 和你一起补齐。") },
        h('div', { className: 'df-skills' }, ...skills.map((skill, i) => h('article', { key: skill.name, className: 'df-skill' + (i === 0 ? ' df-skill-featured' : '') },
          h('div', { className: 'df-skill-heading' }, h('code', null, '/' + skill.name), i === 0 && h('span', { className: 'df-pill' }, t("推荐入口"))),
          h('h3', null, skill.title), h('p', null, skill.detail),
          h('details', null, h('summary', null, t("查看示例")), h(CopyExample, { text: skill.example }))))),
        h('p', { className: 'df-muted' }, t("指令来自当前仓库的 .agents/skills/，共享规则与项目文档按需读取。使用指南只介绍流程，具体方法和参数以项目文件与实际实现为准。"))),
      h(Section, { id: 'results', number: '04', title: t("结果放在任务目录，下一次从这里继续") },
        h('div', { className: 'df-results' }, h('code', null, 'outputs/tasks/<task>/'), h('p', null, t("README.md 是任务入口；task.json 记录阶段、决定、授权与待决定项。Agent 返回实际配置、日志、模型或报告路径。聊天历史保留对话，任务目录保留执行事实。"))),
        h(CopyExample, { text: resumeExample, label: t("复制接续请求") }),
        h('p', { className: 'df-muted' }, t("暂停时保存状态；停止生成只中断当前 Agent 轮次，后台作业是否停止需单独核查。重开会话先检查进程、manifest 和产物，避免重复提交。"))),
      h('footer', { className: 'df-footer' }, h('span', null, t("DecisionForge 插件 · 基于 DeepSeek Harness")), h('span', null, t("完整启动与排查：仓库 docs/zh/agent.md")))));
}

export function apply(ctx) {
  const localized = Component => function Localized(props) {
    const snapshot = React.useSyncExternalStore(fn => ctx.locale.subscribe(fn), () => ctx.locale.getSnapshot(), () => ctx.locale.getSnapshot());
    return h(Language.Provider, { value: snapshot.active === "en" ? "en" : "zh" }, h(Component, props));
  };
  ctx.effect(() => selectLaunchWorkspace(ctx));
  const onChat = () => ctx.layout.selectPanel(null);
  const onGuide = () => ctx.layout.selectPanel(PANEL);
  function HeroMark(props) {
    const t = useCopy();
    return h('span', { className: 'df-hero-brand' },
      h('button', { className: 'df-hero-mark', type: 'button', title: t("DecisionForge 使用指南"), 'aria-label': t("打开 DecisionForge 使用指南"), onClick: onGuide }, h(BrandMark, props)),
      h('span', { className: 'df-hero-slogan' }, t("每个场景，都值得一个决策模型")));
  }
  ctx.effect(() => {
    const style = document.createElement('style');
    style.textContent = styles;
    document.head.appendChild(style);
    return () => style.remove();
  });
  ctx.slots.inject('main', () => ctx.slots.register({ name: 'main', key: PANEL, inject: () => ({ onChat }) }, localized(Guide)));
  ctx.slots.inject('sidebar.panellist', () => {
    let dispose;
    const register = () => {
      dispose?.();
      dispose = ctx.slots.register({ name: 'sidebar.panellist', id: PANEL, order: -10, label: ctx.locale.getSnapshot().active === 'en' ? 'DecisionForge guide' : 'DecisionForge 使用指南' }, GuideIcon);
    };
    register();
    const unsubscribe = ctx.locale.subscribe(register);
    return () => { unsubscribe(); dispose(); };
  });
  ctx.slots.inject('sidebar.brand.mark', () => ctx.slots.register({ name: 'sidebar.brand.mark' }, BrandMark));
  ctx.slots.inject('sidebar.brand.name', () => ctx.slots.register({ name: 'sidebar.brand.name' }, BrandName));
  ctx.slots.inject('conversation.hero.brand.mark', () => ctx.slots.register({ name: 'conversation.hero.brand.mark' }, localized(HeroMark)));
  ctx.effect(() => () => {
    if (ctx.layout.panelInfo.getSnapshot().activePanelId === PANEL) ctx.layout.selectPanel(null);
  });
}
