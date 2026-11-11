import { test } from 'node:test';
import assert from 'node:assert/strict';
import { selectLaunchWorkspace } from '../src/launch-workspace.js';

const tick = () => new Promise(resolve => setImmediate(resolve));
function store(initial) {
  let value = initial;
  const listeners = new Set();
  return {
    getSnapshot: () => value,
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    set(next) { value = next; for (const fn of listeners) fn(); },
    listeners,
  };
}
function runtime() {
  const workspaces = store({ phase: 'loading', items: [] });
  const sessions = store({ phase: 'loading' });
  const opened = [], errors = [];
  const interactionTarget = new EventTarget();
  const ctx = { workspaces: { list: workspaces }, sessions: { list: sessions }, uiWorkspace: { async openWorkspace(id, beforeOpen) { beforeOpen?.('blank'); opened.push(id); } } };
  const options = { interactionTarget, onError: e => errors.push(e), fetchTarget: async () => ({ workspace: { workspaceId: 'checkout' } }) };
  const ready = () => { workspaces.set({ phase: 'ready', items: [{ workspaceId: 'checkout' }] }); sessions.set({ phase: 'ready' }); };
  return { ctx, options, workspaces, sessions, opened, errors, interactionTarget, ready };
}

test('waits for both native lists and target membership, then supersedes old selection once', async () => {
  const r = runtime();
  const dispose = selectLaunchWorkspace(r.ctx, r.options);
  await tick();
  assert.deepEqual(r.opened, []);
  r.sessions.set({ phase: 'ready' });
  r.workspaces.set({ phase: 'ready', items: [] });
  assert.deepEqual(r.opened, []);
  r.ready();
  await tick();
  assert.deepEqual(r.opened, ['checkout']);
  r.ready();
  assert.deepEqual(r.opened, ['checkout']);
  assert.equal(r.workspaces.listeners.size + r.sessions.listeners.size, 0);
  dispose();
});

test('standalone dsh has no project preference and a late fetch cannot override user input', async () => {
  for (const mode of ['standalone', 'disposed', 'interaction']) {
    const r = runtime(); r.ready();
    let resolve;
    r.options.fetchTarget = () => new Promise(done => { resolve = done; });
    const dispose = selectLaunchWorkspace(r.ctx, r.options);
    if (mode === 'disposed') dispose();
    if (mode === 'interaction') r.interactionTarget.dispatchEvent(new Event('pointerdown'));
    resolve({ workspace: mode === 'standalone' ? null : { workspaceId: 'checkout' } });
    await tick();
    assert.deepEqual(r.opened, [], mode);
    assert.equal(r.workspaces.listeners.size + r.sessions.listeners.size, 0);
    dispose();
  }
});

test('unloading while native connection is pending prevents late navigation', async () => {
  const r = runtime(); r.ready();
  let finish;
  r.ctx.uiWorkspace.openWorkspace = async (id, beforeOpen) => {
    await new Promise(resolve => { finish = resolve; });
    beforeOpen('blank');
    r.opened.push(id);
  };
  const dispose = selectLaunchWorkspace(r.ctx, r.options);
  await tick();
  dispose(); finish();
  await tick();
  assert.deepEqual(r.opened, []);
  assert.deepEqual(r.errors, []);
});

test('fetch failure is reported and listeners are released', async () => {
  const r = runtime();
  r.options.fetchTarget = async () => { throw new Error('unavailable'); };
  const dispose = selectLaunchWorkspace(r.ctx, r.options);
  await tick();
  assert.match(r.errors[0].message, /unavailable/);
  assert.equal(r.workspaces.listeners.size + r.sessions.listeners.size, 0);
  dispose();
});
