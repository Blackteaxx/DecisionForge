import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm, realpath } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { apply } from '../src/index.js';

// The actual native workspaceRegistry.create owns persistence and canonical-path reuse.
test('Host registers only an explicit project launch and serves its canonical workspace', async () => {
  const previous = process.env.DFORGE_WORKSPACE_ROOT;
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'dforge launch '));
  const calls = [], routes = [], byPath = new Map();
  const ctx = {
    effect: fn => fn(),
    connection: { fetch: { register(route) { routes.push(route); return () => {}; } } },
    workspaceRegistry: { async create(root, title) {
      calls.push(root);
      if (!byPath.has(root)) byPath.set(root, { id: 'checkout-id', path: root, title });
      return byPath.get(root);
    } },
  };
  const target = async () => (await routes.at(-1).fetch(new Request('http://localhost/api/dforge/launch-workspace'))).json();
  try {
    delete process.env.DFORGE_WORKSPACE_ROOT;
    await apply(ctx);
    assert.deepEqual(await target(), { workspace: null });
    assert.equal(calls.length, 0);
    process.env.DFORGE_WORKSPACE_ROOT = tmp;
    await assert.rejects(() => apply(ctx), /DecisionForge/);
    await mkdir(path.join(tmp, '.agents/skills/dforge'), { recursive: true });
    await writeFile(path.join(tmp, '.agents/skills/dforge/SKILL.md'), 'fixture');
    await writeFile(path.join(tmp, 'README.md'), 'fixture');
    await apply(ctx);
    await apply(ctx);
    assert.deepEqual(await target(), { workspace: { workspaceId: 'checkout-id', path: await realpath(tmp) } });
    assert.equal(byPath.size, 1);
    assert.equal(routes.at(-1).path, '/api/dforge/launch-workspace');
    const response = await routes.at(-1).fetch(new Request('http://localhost/api/dforge/launch-workspace'));
    assert.equal(response.headers.get('cache-control'), 'no-store');
    process.env.DFORGE_WORKSPACE_ROOT = 'relative/path';
    await assert.rejects(() => apply(ctx), /absolute|绝对/);
  } finally {
    if (previous === undefined) delete process.env.DFORGE_WORKSPACE_ROOT;
    else process.env.DFORGE_WORKSPACE_ROOT = previous;
    await rm(tmp, { recursive: true, force: true });
  }
});
