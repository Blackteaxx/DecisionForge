import { access, realpath } from 'node:fs/promises';
import path from 'node:path';

export const inject = ['workspaceRegistry', 'connection'];

export async function apply(ctx) {
  // Only the project launcher supplies this preference. A standalone dsh keeps its own selection.
  const requested = process.env.DFORGE_WORKSPACE_ROOT;
  let workspace = null;
  if (requested) {
    if (!path.isAbsolute(requested)) throw new Error('DecisionForge workspace must be an absolute path.');
    let root;
    try {
      root = await realpath(requested);
      await access(path.join(root, 'README.md'));
      await access(path.join(root, '.agents/skills/dforge/SKILL.md'));
    } catch (cause) {
      throw new Error(`DecisionForge checkout is incomplete or inaccessible: ${requested}`, { cause });
    }
    // The native registry canonicalizes paths and reuses the existing durable workspace identity.
    const registered = await ctx.workspaceRegistry.create(root);
    workspace = { workspaceId: registered.id, path: registered.path };
  }
  // Native exact fetch routes enforce owner authentication, including for this metadata.
  ctx.effect(() => ctx.connection.fetch.register({
    path: '/api/dforge/launch-workspace',
    methods: ['GET'],
    requestBody: 'buffered',
    fetch: async () => Response.json({ workspace }, { headers: { 'Cache-Control': 'no-store' } }),
  }));
}
