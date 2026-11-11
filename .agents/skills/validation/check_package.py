"""Check repository skill links/templates; optionally exercise existing offline CLI paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

import yaml

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / ".agents/skills"
CLAUDE = ROOT / ".claude/skills"
NAMES = {"dforge", "dforge-data", "dforge-train", "dforge-eval", "dforge-analyze"}


def check_package():
    skills = list(PACKAGE.glob("*/SKILL.md"))
    assert {p.parent.name for p in skills} == NAMES, "Expected the five documented skills"
    claude_skills = list(CLAUDE.glob("*/SKILL.md"))
    assert {p.parent.name for p in claude_skills} == NAMES, "Expected Claude discovery entries"
    for name in NAMES | {"shared"}:
        entry = CLAUDE / name
        assert entry.is_symlink(), f"Claude entry must share canonical source: {entry}"
        assert not entry.readlink().is_absolute(), f"Use portable relative links: {entry}"
        assert entry.resolve() == PACKAGE / name, f"Incorrect Claude target: {entry}"
    for path in skills:
        source = path.read_text(encoding="utf-8")
        parts = source.split("---", 2)
        assert len(parts) == 3 and not parts[0].strip(), f"Missing frontmatter: {path}"
        frontmatter = yaml.safe_load(parts[1])
        assert set(frontmatter) == {"name", "description"}, path
        assert frontmatter["name"] == path.parent.name, path
        assert frontmatter["description"].startswith("Use when "), path
        assert len(parts[1]) <= 1024, path
    markdown = [
        *PACKAGE.rglob("*.md"), *claude_skills,
        *(CLAUDE / "shared").rglob("*.md"), ROOT / "README.md", ROOT / "README.en.md",
    ]
    checked_links = 0
    for path in markdown:
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", path.read_text(encoding="utf-8")):
            if ":" in target or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            if target:
                assert (path.parent / target).exists(), f"Broken link in {path}: {target}"
                checked_links += 1
    for path in (PACKAGE / "shared/templates").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        assert record["schema_version"] == "decisionforge.agent-task.v1", path
        assert record["request"]["status"] in {
            "pending", "fulfilled", "waiting_input", "interrupted", "failed"
        }, path
        assert record["outcome"]["status"] in {"unknown", "met", "not_met", "not_applicable"}, path
        resources = {r["id"] for r in record["resources"]}
        artifacts = {a["id"] for a in record["artifacts"]}
        runs = {r["id"] for r in record["runs"]}
        assert len(resources) == len(record["resources"]), path
        assert len(artifacts) == len(record["artifacts"]), path
        assert len(runs) == len(record["runs"]), path
        for run in record["runs"]:
            assert run["resource_id"] in resources, path
            assert run["config_artifact_id"] in artifacts, path
            for evidence in run["evidence"]:
                assert evidence["artifact_id"] in artifacts, path
        for stage in record["stages"].values():
            assert set(stage["run_ids"]) <= runs, path
            assert set(stage["artifact_ids"]) <= artifacts, path
        for artifact in record["artifacts"]:
            assert artifact["storage"] in {"archived", "external", "pending"}, path
            if artifact["storage"] == "archived":
                location = PurePosixPath(artifact["path"])
                assert not location.is_absolute() and ".." not in location.parts, path
            if artifact["resource_id"] is not None:
                assert artifact["resource_id"] in resources, path
    print(
        f"Package: {len(skills)} Codex skills and matching Claude links; "
        f"{checked_links} local links; JSON templates and references valid"
    )


def cli(command, *args):
    prefix = {"inspect": ["data", "inspect"], "validate": ["data", "validate"], "train": ["train", "validate"], "eval": ["eval", "run"]}[command]
    args = [arg for arg in args if not (command == "train" and arg == "--validate-only")]
    result = subprocess.run(
        ["uv", "run", "dforge", *prefix, *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Offline {command} failed: {result.stderr}")
    return json.loads(result.stdout)


def check_offline_cli():
    # Exercise the local checkout, independent of an installed package version.
    sys.path.insert(0, str(ROOT / "src"))
    from decisionforge.config import load_config as load_generation
    from decisionforge.eval.models import load_config as load_evaluation
    from decisionforge.training.config import load_config as load_training

    sources = [ROOT / name for name in (
        "configs/default.yaml", "examples/training/unsloth/qwen35-4b-smoke.yaml", "configs/eval/demo.yaml"
    )]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    # Resolve macOS /var -> /private/var before checking path containment.
    with tempfile.TemporaryDirectory(prefix="dforge-skills-offline-") as directory:
        task = Path(directory).resolve()
        generation, training, evaluation = [yaml.safe_load(p.read_text()) for p in sources]
        generation["input"]["path"] = str(ROOT / "examples/standard.json")
        generation["llm"]["model"] = "offline-preflight"
        generation["output"]["directory"] = str(task / "data/runs/gen-001/run")
        training["data"]["path"] = str(ROOT / "examples/training/data/typed-decisions-sample.jsonl")
        training["output"]["directory"] = str(task / "training/train-001/run")
        evaluation["data"]["path"] = str(ROOT / "examples/eval/samples.jsonl")
        evaluation["output"] = str(task / "evaluation/eval-001/run")
        for command, config, filename, loader in (
            ("inspect", generation, "data/runs/gen-001/config.yaml", load_generation),
            ("train", training, "training/train-001/config.yaml", load_training),
            ("eval", evaluation, "evaluation/eval-001/config.yaml", load_evaluation),
        ):
            path = task / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            args = ["--config", path]
            if command != "inspect":
                args.append("--validate-only")
            result = cli(command, *args)
            loaded = loader(path)
            output = loaded.output if command == "eval" else loaded.output.directory
            assert Path(output).resolve().is_relative_to(task), (command, output)
            assert Path(output).parent == path.parent and Path(output).name == "run", command
            assert not Path(output).exists(), f"Offline preflight created runtime output: {output}"
            count = result["records"] if command == "inspect" else (
                result["total"]["records"] if command == "train" else result["samples"]
            )
            print(f"Offline {command}: {count} records/samples; output contained in task root")
        validation = cli("validate", "--input", ROOT / "examples/training/data/typed-decisions-sample.jsonl")
        print(f"Offline validate: {validation['valid_records']} records")
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    print("Source example configs unchanged; no generation, inference or GPU training executed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-cli", action="store_true", help="Also run offline CLI preflights")
    args = parser.parse_args()
    check_package()
    if args.offline_cli:
        check_offline_cli()


if __name__ == "__main__":
    main()
