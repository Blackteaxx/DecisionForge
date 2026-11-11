"""Real recorder cancellation owns detached descendants and retains live evidence."""

import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests/acceptance/method_acceptance.py"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux scoped subreaper")


@pytest.fixture
def cleanup_scope():
    # The RED reproducer can orphan grandchildren. Adopt them only in this test
    # process, and reap only the exact fixture PIDs; production ownership is tested
    # before this cleanup. Restore the prior subreaper setting afterward.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    processes, owned = [], []
    yield processes, owned
    try:
        for pid in reversed(owned):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            pending = False
            for pid in owned:
                try:
                    result, _ = os.waitpid(pid, os.WNOHANG)
                    pending |= result == 0
                except ChildProcessError:
                    pending |= Path(f"/proc/{pid}").exists()
            if not pending:
                break
            time.sleep(0.02)
    finally:
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def launch(tmp_path, cleanup_scope, mode):
    processes, owned = cleanup_scope
    project, work = tmp_path / "project", tmp_path / "evidence"
    project.mkdir()
    (project / "fake_cli.py").write_text('''import json,os,signal,subprocess,sys,time
from pathlib import Path
config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text())
ready=Path(config['pidfile']); mode=config['mode']
child=subprocess.Popen([sys.executable,'-u','-c',
 'import os,signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);Path(sys.argv[1]).touch();time.sleep(60)',str(ready)+'.child'],start_new_session=True)
while not Path(str(ready)+'.child').exists():time.sleep(.01)
if mode=='wait':print('partial stdout',flush=True)
print('partial stderr',file=sys.stderr,flush=True)
temporary=Path(str(ready)+'.tmp');temporary.write_text(json.dumps([os.getpid(),child.pid]));temporary.replace(ready)
if mode!='wait':
 print(json.dumps({'status':'fixture'}),flush=True)
 os._exit(int(mode))
time.sleep(60)
''')
    config_path = project / "config.json"
    config_path.write_text(json.dumps({"pidfile": str(tmp_path / "pids.json"), "mode": mode}))
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.test", "commit", "-qm", "fixture"]):
        subprocess.run(["git", "-C", str(project), *args], check=True, capture_output=True)
    harness = '''import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('acceptance',sys.argv[1]);driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
driver.ROOT=Path(sys.argv[2])
driver.CLI_COMMAND=[sys.executable,'-u',str(driver.ROOT/'fake_cli.py')]
sys.argv=['acceptance','record','--work-dir',sys.argv[3],'eval','--config',str(driver.ROOT/'config.json'),'--run-dir',str(driver.ROOT/'run')]
raise SystemExit(driver.main())
'''
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    process = subprocess.Popen(
        [sys.executable, "-c", harness, str(DRIVER), str(project), str(work)],
        stdout=(tmp_path / "recorder.stdout").open("w"), stderr=(tmp_path / "recorder.stderr").open("w"), start_new_session=True,
    )
    processes.extend([process, unrelated])
    pidfile = tmp_path / "pids.json"
    deadline = time.monotonic() + 10
    while not pidfile.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pidfile.exists(), f"fixture did not launch: {process.poll()}; {(tmp_path / 'recorder.stderr').read_text()}"
    owned.extend(json.loads(pidfile.read_text()))
    return process, unrelated, work, owned


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_recorder_signal_retains_evidence_and_reaps_scope(tmp_path, cleanup_scope, signum):
    process, unrelated, work, owned = launch(tmp_path, cleanup_scope, "wait")
    command = work / "cli.command.json"
    launch_record = json.loads(command.read_text()) if command.exists() else None
    if launch_record is not None:
        owned.append(launch_record["supervisor_pid"])
    stdout, stderr = work / "cli.stdout.log", work / "cli.stderr.log"
    live_stdout = stdout.read_text() if stdout.exists() else ""
    live_stderr = stderr.read_text() if stderr.exists() else ""
    assert (work / "input-config.yaml").is_file()
    assert json.loads((work / "configuration.json").read_text())["sha256"]
    assert json.loads((work / "source.json").read_text())["commit"]
    process.send_signal(signum)  # Only the wrapper PID; it must forward and reap.
    process.wait(timeout=10)
    record = json.loads(command.read_text()) if command.exists() else None
    assert launch_record is not None, "command evidence must exist before waiting"
    assert launch_record["status"] == "running" and launch_record["command"]
    assert "partial stdout" in live_stdout and "partial stderr" in live_stderr
    assert process.returncode == 128 + signum
    assert record["status"] == "interrupted" and record["returncode"] == 128 + signum
    assert record["cancellation_signal"] == signum and record["ended_at"]
    assert record["scope_reaped"] is True
    assert "partial stdout" in stdout.read_text() and "partial stderr" in stderr.read_text()
    assert all(not Path(f"/proc/{pid}").exists() for pid in owned), "owned child survived recorder"
    assert unrelated.poll() is None, "unrelated sibling must survive"


@pytest.mark.parametrize("code", [0, 7])
def test_recorder_fast_exit_reaps_detached_child_and_records_outcome(tmp_path, cleanup_scope, code):
    process, unrelated, work, owned = launch(tmp_path, cleanup_scope, str(code))
    process.wait(timeout=10)
    record = json.loads((work / "cli.command.json").read_text())
    owned.append(record["supervisor_pid"])
    assert process.returncode == code and record["returncode"] == code
    assert record["status"] == ("completed" if code == 0 else "failed")
    assert record["scope_reaped"] is True
    assert all(not Path(f"/proc/{pid}").exists() for pid in owned)
    assert unrelated.poll() is None
