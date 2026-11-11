import { access, mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { spawn } from 'node:child_process';

const [major, minor] = process.versions.node.split('.').map(Number);
if (major < 24 || (major === 24 && minor < 9)) {
  console.error('DecisionForge plugin registration requires Node.js >=24.9.0.');
  process.exit(1);
}
const plugin = fileURLToPath(new URL('../', import.meta.url));
const root = fileURLToPath(new URL('../../../', import.meta.url));
async function run(command, args) {
  const child = spawn(command, args, { cwd: root, env: process.env, stdio: 'inherit' });
  const interrupt = () => child.kill('SIGINT'), terminate = () => child.kill('SIGTERM');
  process.on('SIGINT', interrupt);
  process.on('SIGTERM', terminate);
  try {
    await new Promise((resolve, reject) => {
      child.on('error', reject);
      child.on('close', (code, signal) => {
        if (code === 0) resolve();
        else reject(Object.assign(new Error(`${command} exited with ${code ?? signal}`), { exitCode: code ?? (signal === 'SIGINT' ? 130 : 143) }));
      });
    });
  } finally {
    process.off('SIGINT', interrupt);
    process.off('SIGTERM', terminate);
  }
}
try {
  let archive;
  if (process.argv[2]) {
    if (process.argv.length > 3) throw new Error('Usage: register.mjs [path/to/plugin.tgz]');
    archive = path.resolve(process.argv[2]);
  } else {
    const pkg = JSON.parse(await readFile(path.join(plugin, 'package.json'), 'utf8'));
    const dist = path.join(plugin, 'dist');
    await mkdir(dist, { recursive: true });
    await run('npm', ['pack', plugin, '--ignore-scripts', '--pack-destination', dist]);
    archive = path.join(dist, `${pkg.name.replace(/^@/, '').replace('/', '-')}-${pkg.version}.tgz`);
  }
  await access(archive);
  await run('npx', ['--yes', '@deepseek-ai/dsh@0.2.2-rc.1', 'plugin', '--profile', 'web', 'add', archive]);
  console.log('DecisionForge registered in this DSH_HOME web profile. Restart dsh web and open Plugins → DecisionForge.');
} catch (error) {
  console.error(error.message);
  process.exitCode = error.exitCode ?? 1;
}
