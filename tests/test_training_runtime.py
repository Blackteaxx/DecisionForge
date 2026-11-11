from types import SimpleNamespace

from decisionforge.training import runtime


def test_windows_cuda_is_not_blocked_by_platform(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(runtime, "version", lambda _: "test")
    monkeypatch.setattr(
        runtime,
        "_cuda_probe",
        lambda *_: {
            "cuda_available": True,
            "gpu": "RTX 5070",
            "cuda": "13.0",
        },
    )
    result = runtime.capabilities()
    assert result["gpu_runtime"] is True
    assert result["gpu"] == "RTX 5070"


def test_installed_packages_without_working_cuda_do_not_enable_training(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(runtime, "version", lambda _: "test")
    monkeypatch.setattr(runtime, "_cuda_probe", lambda *_: {"cuda_available": False})
    assert runtime.capabilities()["gpu_runtime"] is False


def test_missing_dependencies_skip_cuda_probe(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(runtime, "_cuda_probe", lambda *_: (_ for _ in ()).throw(AssertionError()))
    result = runtime.capabilities()
    assert result["gpu_runtime"] is False
    assert "uv sync" in result["note"]


def test_probe_failure_is_explained_and_cached(monkeypatch):
    runtime._cuda_probe.cache_clear()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout="", stderr="DLL initialization failed")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    try:
        first = runtime._cuda_probe("python", ("a",))
        assert first["cuda_available"] is False
        assert first["error"]
        assert runtime._cuda_probe("python", ("a",)) == first
        assert len(calls) == 1
    finally:
        runtime._cuda_probe.cache_clear()
