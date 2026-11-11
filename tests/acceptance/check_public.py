"""Check public guides, offline command help and branding without network access."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
COMMANDS = [
    (),
    ("data",),
    ("train",),
    ("eval",),
    ("schema",),
    *(("data", name) for name in ("inspect", "parse-pdf", "build", "resume", "validate", "export")),
    ("train", "run"),
    ("train", "validate"),
    ("eval", "run"),
    ("eval", "rescore"),
    ("eval", "list"),
    ("schema", "export"),
    ("doctor",),
]


def check() -> None:
    guides = sorted((ROOT / "docs/zh").glob("*.md"))
    assert len(guides) == 8
    assert {p.name for p in guides} == {p.name for p in (ROOT / "docs/en").glob("*.md")}
    markdown = [
        ROOT / name
        for name in ("README.md", "README.en.md", "CONTRIBUTING.md", "CONTRIBUTING.en.md")
    ]
    markdown += [*guides, *sorted((ROOT / "docs/en").glob("*.md"))]
    for path in markdown:
        source = path.read_text(encoding="utf-8")
        if path.name.startswith("README"):
            assert len(source.splitlines()) < 200, path
        for link in re.findall(r"\]\(([^\n)]+)\)", source):
            parsed = urlsplit(link.strip("<>"))
            if parsed.scheme or not parsed.path:
                continue
            assert (path.parent / unquote(parsed.path)).exists(), (path, link)
        for command in re.findall(r"uv run dforge ([a-z-]+)(?: ([a-z-]+))?", source):
            group, action = command
            if group.startswith("--"):
                continue
            assert group in {"data", "train", "eval", "schema", "doctor"}, (path, command)
            if group != "doctor":
                assert (group, action) in COMMANDS, (path, command)
    for command in COMMANDS:
        result = subprocess.run(
            [sys.executable, "-m", "decisionforge", *command, "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (command, result.stderr)
        assert "Usage" in result.stdout, command
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "decisionforge",
            "data",
            "inspect",
            "--config",
            "configs/offline.yaml",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    brand = ROOT / "docs/assets/brand"
    for asset in (ROOT / "demo/assets/brand").iterdir():
        assert asset.read_bytes() == (brand / asset.name).read_bytes(), asset
    assert (ROOT / "extensions/dsh-decisionforge/assets/icon.png").read_bytes() == (
        brand / "icon-512.png"
    ).read_bytes()
    print(
        f"Public checks passed: {len(markdown)} guides/readmes, {len(COMMANDS)} help commands, offline example and shared brand assets"
    )


if __name__ == "__main__":
    check()
