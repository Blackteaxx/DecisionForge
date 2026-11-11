import { access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 24 || (major === 24 && minor < 9)) {
  console.error('DecisionForge requires Node.js >=24.9.0. Activate a supported Node version first.');
  process.exit(1);
}
const root = path.resolve(fileURLToPath(new URL('../../../', import.meta.url)));
const plugin = fileURLToPath(new URL('../', import.meta.url));
try {
  await access(path.join(plugin, 'lib/client.js'));
  await access(path.join(root, '.agents/skills/dforge/SKILL.md'));
} catch {
  console.error('Run npm ci && npm run build in extensions/dsh-decisionforge inside a complete DecisionForge checkout first.');
  process.exit(1);
}
console.log('DecisionForge · default workspace:', root);
const child = spawn('npx', ['--yes', '@deepseek-ai/dsh@0.2.2-rc.1', 'web', ...process.argv.slice(2)], {
  cwd: root, env: { ...process.env, DFORGE_WORKSPACE_ROOT: root }, stdio: 'inherit',
});
const forward = signal => () => child.kill(signal);
const onInterrupt = forward('SIGINT'), onTerminate = forward('SIGTERM');
process.on('SIGINT', onInterrupt);
process.on('SIGTERM', onTerminate);
let spawnError;
child.on('error', error => { spawnError = error; });
child.on('close', async (code, signal) => {
  process.off('SIGINT', onInterrupt);
  process.off('SIGTERM', onTerminate);
  if (spawnError) console.error('Could not launch the official Harness CLI:', spawnError.message);
  process.exitCode = spawnError ? 1 : code ?? (signal === 'SIGINT' ? 130 : signal === 'SIGTERM' ? 143 : 1);
});
