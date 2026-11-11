async function fetchLaunchWorkspace(signal) {
  // Relative to the app base so a reverse-proxy URL prefix is preserved.
  const response = await fetch('./api/dforge/launch-workspace', { credentials: 'same-origin', signal });
  if (!response.ok) throw new Error(`DecisionForge workspace lookup failed (${response.status})`);
  return response.json();
}

/** Select the launch checkout once, after native restoration has its initial list snapshots. */
export function selectLaunchWorkspace(ctx, {
  fetchTarget = fetchLaunchWorkspace,
  interactionTarget = window,
  onError = error => console.warn('DecisionForge: could not select launch workspace; select it in the workspace menu.', error),
} = {}) {
  const lifetime = new AbortController();
  const subscriptions = [];
  const stopWatching = () => {
    for (const stop of subscriptions.splice(0)) stop();
    interactionTarget.removeEventListener('pointerdown', cancel, true);
    interactionTarget.removeEventListener('keydown', cancel, true);
  };
  const cancel = () => { lifetime.abort(); stopWatching(); };
  // A slow connection must not take the user away from a workspace they just chose.
  interactionTarget.addEventListener('pointerdown', cancel, { capture: true });
  interactionTarget.addEventListener('keydown', cancel, { capture: true });
  let started = false;
  let target;
  const reconcile = () => {
    if (started || lifetime.signal.aborted || !target) return;
    const workspaces = ctx.workspaces.list.getSnapshot();
    if (workspaces.phase !== 'ready' || ctx.sessions.list.getSnapshot().phase !== 'ready') return;
    if (!workspaces.items.some(item => item.workspaceId === target.workspaceId)) return;
    started = true;
    stopWatching();
    // openWorkspace begins native navigation, superseding any pending saved-session restore.
    Promise.resolve().then(() => {
      lifetime.signal.throwIfAborted();
      return ctx.uiWorkspace.openWorkspace(target.workspaceId, () => lifetime.signal.throwIfAborted());
    }).catch(error => { if (!lifetime.signal.aborted) onError(error); });
  };
  // Fetch before subscribing; the native UI service is injected first and watches restoration first.
  fetchTarget(lifetime.signal).then(result => {
    if (lifetime.signal.aborted) return;
    target = result.workspace;
    if (!target) { stopWatching(); return; }
    subscriptions.push(ctx.workspaces.list.subscribe(reconcile), ctx.sessions.list.subscribe(reconcile));
    reconcile();
  }).catch(error => {
    stopWatching();
    if (!lifetime.signal.aborted) onError(error);
  });
  return cancel;
}
