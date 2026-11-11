import { build } from 'esbuild';
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
const pkg = JSON.parse(await readFile(new URL('../package.json', import.meta.url)));
const result = await build({
  absWorkingDir: root, entryPoints: ['src/client.js'], bundle: true, format: 'cjs',
  platform: 'browser', target: 'es2022', external: ['react'], write: false,
  loader: { '.png': 'dataurl', '.css': 'text' }, minify: true,
});
await mkdir(new URL('../lib/', import.meta.url), { recursive: true });
await writeFile(new URL('../lib/client.js', import.meta.url), `window.__ModuleLoader__.load({id:${JSON.stringify(pkg.name)},factory:(require)=>{var module={exports:{}};var exports=module.exports;\n${result.outputFiles[0].text}\nreturn module.exports;}});\n`);
await copyFile(new URL('../src/index.js', import.meta.url), new URL('../lib/index.js', import.meta.url));
await copyFile(new URL('../assets/icon.png', import.meta.url), new URL('../lib/logo.png', import.meta.url));
console.log(`Built ${pkg.name}@${pkg.version} (logo and styles embedded)`);
