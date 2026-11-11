import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, access } from 'node:fs/promises';
const root = new URL('../', import.meta.url);
test('package is a portable native dsh bundle with localized plugin details', async () => {
  const pkg = JSON.parse(await readFile(new URL('package.json', root)));
  assert.equal(pkg.dsh.bundle?.patch, './cordis.patch.yml');
  assert.equal(pkg.exports['./cordis.patch.yml'], './cordis.patch.yml');
  assert.equal(pkg.exports['./locale/*.json'], './locale/*.json');
  const patch = await readFile(new URL(pkg.dsh.bundle.patch, root), 'utf8');
  assert.match(patch, /id: ui-brand-official\s+disabled: true/);
  assert.ok(patch.includes("name: '@decisionforge/dsh-plugin'"));
  assert.ok(!patch.includes('/Users/'));
  for (const language of ['zh', 'en']) {
    const locale = JSON.parse(await readFile(new URL(`locale/${language}.json`, root)));
    assert.match(locale.meta.title, /DecisionForge/);
    await access(new URL(locale.meta.detail, root));
  }
});

test('plugin manager icon fits the native metadata size limit', async () => {
  const pkg = JSON.parse(await readFile(new URL('package.json', root)));
  const icon = await readFile(new URL(pkg.icon, root));
  assert.ok(icon.byteLength <= 256 * 1024, 'native plugin icons must be at most 256 KiB');
});
