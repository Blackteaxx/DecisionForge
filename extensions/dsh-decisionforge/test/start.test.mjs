import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, chmod, readFile, rm, mkdir, copyFile, realpath } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

async function withFakeCli(fn) {
  const tmp = await mkdtemp(path.join(os.tmpdir(), 'dforge-cli-test-'));
  try {
    const capture = path.join(tmp, 'capture.jsonl');
    const executable = path.join(tmp, 'npx');
    await writeFile(executable, `#!${process.execPath}\nconst fs=require('node:fs');fs.appendFileSync(process.env.DFORGE_TEST_CAPTURE,JSON.stringify({args:process.argv.slice(2),cwd:process.cwd(),kept:process.env.DFORGE_TEST_KEEP,dshHome:process.env.DSH_HOME,workspaceRoot:process.env.DFORGE_WORKSPACE_ROOT})+'\\n');process.exit(Number(process.env.DFORGE_TEST_EXIT||0));`);
    await chmod(executable, 0o755);
    const env = { ...process.env, PATH: tmp + path.delimiter + process.env.PATH, DFORGE_TEST_CAPTURE: capture, DFORGE_TEST_KEEP: 'preserved', DSH_HOME: path.join(tmp, 'state') };
    await fn({ tmp, capture, env });
  } finally { await rm(tmp, { recursive: true, force: true }); }
}

test('launcher boots the registered web profile without invocation overlays or package writes', async () => {
  await withFakeCli(async ({ tmp, capture, env }) => {
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('../scripts/start.mjs', import.meta.url)), '--no-open', '--port', '3081'], { cwd: tmp, env, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const calls = (await readFile(capture, 'utf8')).trim().split('\n').map(JSON.parse);
    assert.equal(calls.length, 1);
    const observed = calls[0];
    assert.deepEqual(observed.args, ['--yes', '@deepseek-ai/dsh@0.2.2-rc.1', 'web', '--no-open', '--port', '3081']);
    assert.equal(observed.cwd, fileURLToPath(new URL('../../../', import.meta.url)).replace(/\/$/, ''));
    assert.equal(observed.workspaceRoot, observed.cwd);
    assert.equal(observed.kept, 'preserved');
    assert.equal(observed.dshHome, env.DSH_HOME);
  });
});

test('registration forwards a tarball path with spaces to native dsh package management', async () => {
  await withFakeCli(async ({ tmp, capture, env }) => {
    const archive = path.join(tmp, 'plugin with spaces.tgz');
    await writeFile(archive, 'fixture');
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('../scripts/register.mjs', import.meta.url)), archive], { cwd: tmp, env, encoding: 'utf8' });
    assert.equal(result.status, 0, result.stderr);
    const observed = JSON.parse((await readFile(capture, 'utf8')).trim());
    assert.deepEqual(observed.args, ['--yes', '@deepseek-ai/dsh@0.2.2-rc.1', 'plugin', '--profile', 'web', 'add', archive]);
    assert.equal(observed.dshHome, env.DSH_HOME);
  });
});

test('registration preserves package manager failure status', async () => {
  await withFakeCli(async ({ tmp, env }) => {
    const archive = path.join(tmp, 'plugin.tgz');
    await writeFile(archive, 'fixture');
    const result = spawnSync(process.execPath, [fileURLToPath(new URL('../scripts/register.mjs', import.meta.url)), archive], { cwd: tmp, env: { ...env, DFORGE_TEST_EXIT: '7' }, encoding: 'utf8' });
    assert.equal(result.status, 7, result.stderr);
  });
});


test('launcher resolves a relocated checkout with spaces and overrides a stale launch root', async () => {
  await withFakeCli(async ({ tmp, capture, env }) => {
    const root = path.join(tmp, 'cloned DecisionForge project');
    const extension = path.join(root, 'extensions/dsh-decisionforge');
    await mkdir(path.join(extension, 'scripts'), { recursive: true });
    await mkdir(path.join(extension, 'lib'));
    await mkdir(path.join(root, '.agents/skills/dforge'), { recursive: true });
    await copyFile(new URL('../scripts/start.mjs', import.meta.url), path.join(extension, 'scripts/start.mjs'));
    await writeFile(path.join(extension, 'lib/client.js'), '');
    await writeFile(path.join(root, '.agents/skills/dforge/SKILL.md'), 'fixture');
    const result = spawnSync(process.execPath, [path.join(extension, 'scripts/start.mjs'), '--no-open'], {
      cwd: tmp, env: { ...env, DFORGE_WORKSPACE_ROOT: '/stale/other-checkout' }, encoding: 'utf8',
    });
    assert.equal(result.status, 0, result.stderr);
    const observed = JSON.parse((await readFile(capture, 'utf8')).trim());
    assert.equal(observed.cwd, await realpath(root));
    assert.equal(await realpath(observed.workspaceRoot), await realpath(root));
  });
});
