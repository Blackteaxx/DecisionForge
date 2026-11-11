import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderToStaticMarkup } from 'react-dom/server';
import { createElement } from 'react';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

async function load() {
  let snapshot = { active: "zh" };
  const listeners = new Set();
  const registrations = [], disposers = [], selected = [];
  const document = {
    head: { appendChild() {} },
    createElement() { return { textContent: '', remove() {} }; },
  };
  let plugin;
  const react = await import('react');
  vm.runInNewContext(await readFile(new URL('../lib/client.js', import.meta.url), 'utf8'), {
    window: { addEventListener() {}, removeEventListener() {}, __ModuleLoader__: { load({ factory }) { plugin = factory(name => {
      assert.equal(name, 'react'); return react;
    }); } } }, document, console, AbortController, fetch: async () => ({ ok: true, json: async () => ({ workspace: null }) }),
  });
  const ctx = {
    effect(fn) { disposers.push(fn()); },
    locale: { getSnapshot: () => snapshot, subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); } },
    layout: { panelInfo: { getSnapshot: () => ({ activePanelId: selected.at(-1) ?? null }) }, selectPanel(id) { selected.push(id); } },
    slots: {
      inject(name, fn) {
        const result = fn();
        if (result?.next) for (const disposer of result) disposers.push(disposer);
        else if (typeof result === 'function') disposers.push(result);
        return () => {};
      },
      register(options, component) {
        const r = { options, component }; registrations.push(r);
        return () => registrations.splice(registrations.indexOf(r), 1);
      },
    },
  };
  plugin.apply(ctx);
  return { registrations, selected, setLocale(active) { snapshot = { active }; for (const fn of [...listeners]) fn(); }, dispose() { for (const fn of disposers.reverse()) fn?.(); } };
}

test('native brand slots and guide panel unload together', async () => {
  const runtime = await load();
  const entries = runtime.registrations;
  for (const name of ['sidebar.brand.mark', 'sidebar.brand.name', 'conversation.hero.brand.mark', 'main', 'sidebar.panellist']) {
    assert.ok(entries.some(r => r.options.name === name), name);
  }
  const main = entries.find(r => r.options.name === 'main');
  const nav = entries.find(r => r.options.name === 'sidebar.panellist');
  assert.equal(main.options.key, nav.options.id);
  runtime.dispose();
  assert.equal(entries.length, 0);
});

test('guide describes five repository skills and confirmation boundaries', async () => {
  const runtime = await load();
  const guide = runtime.registrations.find(r => r.options.name === 'main');
  const html = renderToStaticMarkup(createElement(guide.component, guide.options.inject?.()));
  for (const skill of ['dforge', 'dforge-data', 'dforge-train', 'dforge-eval', 'dforge-analyze']) assert.ok(html.includes('/' + skill));
  for (const phrase of ['选择工作区', '接受', '修改', '暂停', '总体路线', 'outputs/tasks/', 'task.json']) assert.ok(html.includes(phrase), phrase);
  assert.ok(!html.includes('<script'));
  runtime.dispose();
});

test('host locale switches guide, hero and sidebar together', async () => {
  const runtime = await load();
  const render = name => {
    const entry = runtime.registrations.find(r => r.options.name === name);
    return renderToStaticMarkup(createElement(entry.component, entry.options.inject?.()));
  };
  assert.match(render('conversation.hero.brand.mark'), /每个场景，都值得一个决策模型/);
  runtime.setLocale('en');
  const english = render('main');
  assert.match(english, /Choose a workspace/);
  assert.match(english, /dforge-data/);
  assert.doesNotMatch(english, /[\u4e00-\u9fff]/);
  assert.match(render('conversation.hero.brand.mark'), /Every scenario deserves its own decision model/);
  assert.equal(runtime.registrations.find(r => r.options.name === 'sidebar.panellist').options.label, 'DecisionForge guide');
  runtime.setLocale('zh');
  assert.match(render('main'), /选择工作区/);
  runtime.dispose();
});
