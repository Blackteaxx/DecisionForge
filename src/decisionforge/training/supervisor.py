"""Dedicated Linux ownership boundary for one training launch; imports no GPU SDK."""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import time
from pathlib import Path

PR_SET_PDEATHSIG = 1
PR_SET_CHILD_SUBREAPER = 36


def _identity(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _descendants():
    # Unlike the optional /proc/task/children interface, stat is available on
    # the supported hosts. Reparented orphans remain children of this subreaper.
    parents = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(") ", 1)[1].split()
            parents.setdefault(int(fields[1]), []).append((int(stat.parent.name), fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    found, pending = {}, [os.getpid()]
    while pending:
        for pid, identity in parents.get(pending.pop(), ()):
            if pid not in found:
                found[pid] = identity
                pending.append(pid)
    return found


def _reap(launcher_pid, launcher_code):
    while True:
        try:
            pid, status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return launcher_code, False
        if pid == 0:
            return launcher_code, True
        if pid == launcher_pid:
            launcher_code = os.waitstatus_to_exitcode(status)


def _drain(launcher_pid, launcher_code):
    deadline, terminated = time.monotonic() + 2.0, set()
    while True:
        owned = _descendants()
        for pid, identity in owned.items():
            if _identity(pid) != identity:
                continue
            if time.monotonic() >= deadline:
                signum = signal.SIGKILL
            elif (pid, identity) not in terminated:
                signum = signal.SIGTERM
                terminated.add((pid, identity))
            else:
                continue
            try:
                os.kill(pid, signum)
            except ProcessLookupError:
                pass
        launcher_code, pending = _reap(launcher_pid, launcher_code)
        if not pending:
            # ECHILD is the completion condition, not an empty sampled PID list.
            # All fast-exiting parents reparent their children here before reaping.
            return launcher_code
        time.sleep(0.02)


def supervise(command, parent_pid):
    stopped = None

    def stop(signum, frame):
        nonlocal stopped
        stopped = signum

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, stop)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, *([ctypes.c_ulong] * 4)]
    libc.prctl.restype = ctypes.c_int
    # Establish ownership before launching anything. This applies only to this
    # dedicated process, not the coordinator or its unrelated children.
    for option, value in ((PR_SET_CHILD_SUBREAPER, 1), (PR_SET_PDEATHSIG, signal.SIGTERM)):
        if libc.prctl(option, value, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, "Cannot establish training process supervision")
    if os.getppid() != parent_pid or stopped is not None:
        return 128 + (stopped or signal.SIGTERM)
    launcher = subprocess.Popen(command)
    code = None
    try:
        while code is None and stopped is None:
            code, _ = _reap(launcher.pid, code)
            if code is None and stopped is None:
                time.sleep(0.02)
    finally:
        # Never exit while a worker still belongs to this scope. SIGKILL is
        # escalated inside the scope; the coordinator never kills its subreaper.
        code = _drain(launcher.pid, code)
        launcher.returncode = code
    if stopped is not None:
        return 128 + stopped
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Internal training process supervisor")
    parser.add_argument("--parent-pid", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a launcher command is required")
    raise SystemExit(supervise(command, args.parent_pid))
