"""Freeze NLM-owned summaries and assign roles before constructing any patients."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path


class SummaryText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "li", "br", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "li", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def load_nlm_topics(xml_path: Path) -> list[dict]:
    root = ET.parse(xml_path).getroot()
    topics = []
    for topic in root.findall("health-topic"):
        if topic.get("language") != "English":
            continue
        summary = topic.find("full-summary")
        if summary is None:
            continue
        parser = SummaryText()
        parser.feed("".join(summary.itertext()))
        text = "\n".join(
            " ".join(line.split()) for line in "".join(parser.parts).splitlines() if line.strip()
        )
        if not text:
            continue
        topics.append(
            {
                "id": topic.attrib["id"],
                "family_id": topic.attrib["id"],
                "title": topic.attrib["title"],
                "url": topic.attrib["url"],
                "summary": text,
                "summary_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "attribution": "Source: MedlinePlus, National Library of Medicine.",
                "rights": "NLM-owned health-topic summary; public domain",
                "source_kind": "NLM_health_topic_full_summary",
            }
        )
    return topics


def assign_source_roles(topics: list[dict], seed: int = 3407) -> list[dict]:
    result = []
    for topic in topics:
        family = str(topic.get("family_id", topic["id"]))
        value = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest()[:12], 16) % 10
        role = "train" if value < 8 else "validation" if value == 8 else "calibration"
        result.append({**topic, "family_id": family, "split": role, "split_seed": seed})
    return result


def write_materials(topics: list[dict], directory: Path):
    """Write one unmodified source summary per TXT, keeping its assigned role."""
    for topic in topics:
        path = directory / topic["split"] / f"{topic['id']}.txt"
        content = f"{topic['title']}\n\n{topic['summary']}\n"
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Existing source material differs: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--text-directory", type=Path,
        help="Also write train/validation/calibration TXT materials for native build",
    )
    args = parser.parse_args()
    topics = assign_source_roles(load_nlm_topics(args.xml), args.seed)
    pattern = re.compile(
        r"\b(emergency|urgent|call 911|call your|see a doctor|see your doctor|"
        r"contact your|health care provider|healthcare provider|seek medical|"
        r"at home|on (?:its|their) own|self-care)\b",
        re.I,
    )
    candidates = [
        {
            **t,
            "candidate_rule_spans": [p for p in t["summary"].splitlines() if pattern.search(p)],
            "admission_status": "candidate_requires_teacher_check",
        }
        for t in topics
        if pattern.search(t["summary"])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in candidates))
    if args.text_directory is not None:
        write_materials(candidates, args.text_directory)
    from collections import Counter

    print(
        json.dumps(
            {
                "english_topics": len(topics),
                "action_candidates": len(candidates),
                "roles": dict(Counter(t["split"] for t in candidates)),
                "xml_sha256": hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                "output": str(args.output.resolve()),
                "status": "source_candidates_not_accepted_training_data",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
