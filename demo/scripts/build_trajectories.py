#!/usr/bin/env python3
"""Convert exported agent sessions into the playback data used by the demo page.

Usage:
    python demo/scripts/build_trajectories.py <trajectory-export-dir>

The export directory is the unpacked ``trajectory-export-*.tar.gz`` (it contains
``manifest.json``, ``triage-codex/`` and ``materials-claude/``). The script writes
``demo/data/trajectory-<id>.js``; nothing in the export is modified.

Each event is tagged with a node of the DecisionForge skills flow (see
``docs/zh/agent.md``) using the hand-curated time breakpoints in
``TRAJECTORIES``. Paths, user names, IP addresses and credential-like strings are
redacted before anything is written.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[1] / "data"

CMD_LIMIT = 420
OUT_LIMIT = 420
GROUP_LIMIT = 30

# --------------------------------------------------------------------------------------
# Curation: which sessions belong to a trajectory, and which skills-flow node each time
# window belongs to. Breakpoints are UTC ISO prefixes; an event belongs to the last
# breakpoint whose time is <= the event time. ``round`` > 1 marks an iteration loop.
# --------------------------------------------------------------------------------------
TRAJECTORIES = {
    "triage": {
        "group": "triage-codex",
        "harness": "Codex",
        "sessions": [
            ("rollout-2026-10-28T12-59-58-01a180e1-189f-7771-a570-0ac23b1acf57.jsonl", "会话 1 · 定义需求，创建 Issue #22"),
            ("rollout-2026-10-28T13-26-55-01a180f9-c8e2-77c0-980d-edbd5af8fca9.jsonl", "会话 2 · 执行 Issue #22：数据、训练、评测与优化"),
        ],
        "subagents": {
            "review_triage_issue": "rollout-2026-10-28T13-22-05-01a180f5-5bfe-79a3-b46d-9860e8a8d728.jsonl",
            "design_review": "rollout-2026-10-28T13-32-56-01a180ff-47b8-7021-8db4-6ef11f9d69e2.jsonl",
            "implementation_review": "rollout-2026-10-28T16-05-18-01a1818a-c866-7b32-991d-2465ebfe62f9.jsonl",
            "optimization_review": "rollout-2026-10-28T20-24-03-01a18277-ab04-76d1-a549-e17633d5f00c.jsonl",
        },
        "breakpoints": [
            ("2026-10-28T05:17:00", "request", 1),
            ("2026-10-28T05:17:06", "clarify", 1),
            ("2026-10-28T05:20:00", "roadmap", 1),
            ("2026-10-28T05:34:30", "ownEval", 1),
            ("2026-10-28T05:41:30", "ownSearch", 1),
            ("2026-10-28T05:46:00", "ownTrainData", 1),
            ("2026-10-28T06:49:00", "ownEval", 1),
            ("2026-10-28T07:04:30", "ownTrainData", 1),
            ("2026-10-28T07:48:30", "train", 1),
            ("2026-10-28T08:52:00", "evaluation", 1),
            ("2026-10-28T10:23:00", "report", 1),
            ("2026-10-28T10:59:00", "ownTrainData", 2),
            ("2026-10-28T14:15:30", "train", 2),
            ("2026-10-28T16:24:00", "evaluation", 2),
            ("2026-10-28T16:40:00", "report", 2),
            ("2026-10-29T02:17:00", "delivery", 1),
        ],
    },
    "materials": {
        "group": "materials-claude",
        "harness": "Claude Code",
        "sessions": [
            ("fd9944e5-925c-4b21-98df-e5dd0db7fd57.jsonl", "会话 1 · 定义需求，创建 Issue #25"),
            ("934f6efd-622f-4f93-b1ba-63bddb277e7e.jsonl", "会话 2 · 执行 Issue #25：评测集、数据、训练、评测与扩展"),
        ],
        "subagents": {},
        "breakpoints": [
            ("2026-10-29T08:18:00", "request", 1),
            ("2026-10-29T08:18:15", "clarify", 1),
            ("2026-10-29T08:31:00", "roadmap", 1),
            ("2026-10-29T09:16:00", "ownEval", 1),
            ("2026-10-29T09:25:00", "ownSearch", 1),
            ("2026-10-29T09:36:20", "ownTrainData", 1),
            ("2026-10-29T09:38:30", "ownEval", 1),
            ("2026-10-29T09:39:25", "ownTrainData", 1),
            ("2026-10-29T10:03:00", "evaluation", 1),
            ("2026-10-29T10:13:00", "ownTrainData", 1),
            ("2026-10-29T10:36:00", "train", 1),
            ("2026-10-29T14:49:00", "evaluation", 1),
            ("2026-10-29T15:42:00", "report", 1),
            ("2026-10-30T02:14:00", "delivery", 1),
            ("2026-11-02T07:38:00", "ownEval", 2),
            ("2026-11-02T08:24:00", "evaluation", 2),
            ("2026-11-02T11:24:00", "report", 2),
            ("2026-11-03T07:06:00", "delivery", 2),
        ],
    },
}

# --------------------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------------------
PATH_RULES = [
    ("/mnt/shared-storage-user/hutu/DecisionForge-data/", ""),
    ("/mnt/shared-storage-user/hutu/DecisionForge-data", "."),
    ("/mnt/shared-storage-gpfs2/ai4energy/hutu/", "$STORE/"),
    ("/mnt/shared-storage-gpfs2/ai4energy/", "$GPFS/"),
    ("/mnt/shared-storage-gpfs2/", "$GPFS/"),
    ("/mnt/shared-storage-user/hutu/", "$SHARED/"),
    ("/mnt/shared-storage-user/", "$SHARED/"),
    ("/mnt/shared-storage-user", "$SHARED"),
    ("/mnt/", "$MNT/"),
    ("/root/", "~/"),
]
SECRET_RULES = [
    (re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9_\-]{12,}"), "<redacted-key>"),
    (re.compile(r"\b[\w-]+(?:\.[\w-]+)*\.pjlab\.(?:org\.cn|local)\b"), "<host>"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{12,}"), "Bearer <redacted>"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret|password)[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9._\-]{12,})"), r"\1<redacted>"),
    (re.compile(r"gAAAAAB[A-Za-z0-9_\-=]{20,}"), "<encrypted>"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
    (re.compile(r"\b(?!127\.0\.0\.1\b)(?!0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"(?<=@)<ip>"), "<ip>"),
    (re.compile(r"(?i)boyue"), "provider"),
    (re.compile(r"\bhutu\b|hutu(?=[-_/])|(?<=[-_/])hutu"), "user"),
]


def redact(text: str | None) -> str:
    if not text:
        return ""
    for old, new in PATH_RULES:
        text = text.replace(old, new)
    for pattern, repl in SECRET_RULES:
        text = pattern.sub(repl, text)
    return text


def clip(text: str, limit: int) -> str:
    text = text.strip("\n")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f" …（共 {len(text):,} 字符）"


def ts_key(ts: str) -> str:
    return ts[:19]


# --------------------------------------------------------------------------------------
# Parsers → raw events: dict(t, k, ...)
#   k: user | assistant | ask | tool | note | subagent
# --------------------------------------------------------------------------------------
JS_STRING = re.compile(r'cmd\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def codex_tool(name: str, raw: str) -> tuple[str, str, str]:
    """Return (tool name, title, body) for a Codex tool call."""
    if name == "exec":
        cmds = []
        for m in JS_STRING.findall(raw or ""):
            try:
                cmds.append(json.loads('"' + m + '"'))
            except Exception:
                cmds.append(m)
        if cmds:
            body = "\n∥\n".join(cmds)
            return "shell", command_title(cmds[0]) + (f"  (+{len(cmds) - 1} 并行)" if len(cmds) > 1 else ""), body
        inner = re.findall(r"tools\.(\w+)\(", raw or "")
        label = inner[0] if inner else "exec"
        if label == "apply_patch":
            files = re.findall(r"\*\*\* (?:Update|Add|Delete) File: ([^\n\\]+)", raw)
            return "apply_patch", "编辑 " + ", ".join(f.split("/")[-1] for f in files[:3]) if files else "apply_patch", raw
        if label == "write_stdin":
            return "stdin", "读取后台进程输出", raw
        if label.startswith("web"):
            q = re.findall(r'q(?:uery)?\s*:\s*"([^"]+)"', raw)
            return "web", "检索 " + (q[0] if q else "网页"), raw
        if label.startswith("mcp__codex_apps__github"):
            return "github", label.replace("mcp__codex_apps__", ""), raw
        return label, label, raw
    if name == "exec_command":
        try:
            cmd = json.loads(raw or "{}").get("cmd", "")
        except Exception:
            cmd = raw or ""
        return "shell", command_title(cmd), cmd
    if name == "sleep":
        try:
            ms = json.loads(raw or "{}").get("duration_ms", 0)
        except Exception:
            ms = 0
        return "wait", f"等待 {round(ms / 1000)} 秒", ""
    return name, name, raw or ""


def command_title(cmd: str) -> str:
    lines = [ln.strip() for ln in cmd.strip().splitlines() if ln.strip()]
    skip = re.compile(r"^(set -[euxo]+|cd \S+|export \S+=\S+|source \S+)$")
    lines = [ln for ln in lines if not skip.match(ln)] or lines or [""]
    first = lines[0]
    if re.match(r"^(\S*python3?|\S*/python)\s+-\s*<<", first) and len(lines) > 1:
        body = [ln for ln in lines[1:] if not ln.startswith(("from ", "import ")) and ln not in ("PY", "EOF")]
        first = "python ▸ " + (body[0] if body else lines[1])
    return first[:160]


def strip_codex_output(text: str) -> str:
    text = re.sub(r"^Script completed\nWall time [^\n]*\nOutput:\n", "", text or "")
    text = re.sub(r"^Chunk ID: \w+\nWall time: [^\n]*\n(Process exited with code \d+\n)?(Original token count: \d+\n)?Output:\n", "", text)
    return text


def parse_codex(path: Path) -> list[dict]:
    events: list[dict] = []
    calls: dict[str, dict] = {}
    for line in path.open():
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp", "")
        pl = o.get("payload") or {}
        typ = o.get("type")
        if typ == "event_msg" and pl.get("type") == "item_completed":
            item = pl.get("item") or {}
            if item.get("type") != "UserMessage":
                continue
            text = "".join(c.get("text", "") for c in item.get("content", []) if isinstance(c, dict))
            if text.startswith("<send_user_message_question_reply>"):
                body = text.split(">", 1)[1].rsplit("</send_user_message_question_reply>", 1)[0]
                try:
                    answers = json.loads(body)
                except Exception:
                    answers = []
                events.append(dict(t=ts, k="answer", answers=[(a.get("question", ""), a.get("answer", "")) for a in answers]))
            else:
                events.append(dict(t=ts, k="user", x=text))
            continue
        if typ != "response_item":
            continue
        t = pl.get("type")
        if t == "message" and pl.get("role") == "assistant":
            text = "".join(c.get("text", "") for c in pl.get("content", []) if isinstance(c, dict))
            if text.strip():
                events.append(dict(t=ts, k="assistant", x=text))
        elif t in ("custom_tool_call", "function_call"):
            name = pl.get("name")
            raw = pl.get("input") if t == "custom_tool_call" else pl.get("arguments")
            if name == "request_user_input_async":
                try:
                    qs = json.loads(raw).get("questions", [])
                except Exception:
                    qs = []
                events.append(dict(t=ts, k="ask", questions=[dict(q=q.get("title", ""), options=q.get("options") or []) for q in qs], call=pl.get("call_id")))
                continue
            if name == "spawn_agent":
                try:
                    a = json.loads(raw)
                except Exception:
                    a = {}
                events.append(dict(t=ts, k="subagent", name=a.get("task_name", "subagent"), model=a.get("model", "")))
                continue
            if name in ("send_message", "list_agents"):
                continue
            tool, title, body = codex_tool(name, raw)
            ev = dict(t=ts, k="tool", n=tool, c=title, b=body, o="", e=False)
            calls[pl.get("call_id")] = ev
            events.append(ev)
        elif t in ("custom_tool_call_output", "function_call_output"):
            ev = calls.get(pl.get("call_id"))
            if ev is None:
                continue
            out = pl.get("output")
            if isinstance(out, list):
                out = "\n".join(x.get("text", "") for x in out if isinstance(x, dict))
            elif isinstance(out, dict):
                out = json.dumps(out, ensure_ascii=False)
            out = strip_codex_output(out or "")
            ev["o"] = out
            ev["e"] = bool(re.search(r"Process exited with code [1-9]|Traceback \(most recent|\bError:", out[:2000]))
    return events


def codex_subagent_summary(path: Path) -> str:
    texts = []
    for line in path.open():
        o = json.loads(line)
        pl = o.get("payload") or {}
        if o.get("type") == "response_item" and pl.get("type") == "message" and pl.get("role") == "assistant":
            texts.append("".join(c.get("text", "") for c in pl.get("content", []) if isinstance(c, dict)))
    texts = [t for t in texts if len(t) > 60]
    return texts[-1] if texts else ""


CLAUDE_ANSWER = re.compile(r'"([^"]+)"="([^"]+)"')


def claude_tool(name: str, inp: dict) -> tuple[str, str]:
    if name == "Bash":
        return inp.get("description") or command_title(inp.get("command", "")), inp.get("command", "")
    if name in ("Read", "Write", "Edit"):
        p = inp.get("file_path", "")
        verb = {"Read": "读取", "Write": "写入", "Edit": "编辑"}[name]
        return f"{verb} {p.split('/')[-1]}", p
    if name == "WebSearch":
        return "检索 " + inp.get("query", ""), inp.get("query", "")
    if name == "WebFetch":
        return "抓取 " + inp.get("url", ""), inp.get("prompt", "")
    if name == "Monitor":
        return inp.get("description") or "挂起后台监控", inp.get("command", "") or json.dumps(inp, ensure_ascii=False)
    if name == "Skill":
        return "加载 skill: " + inp.get("skill", ""), inp.get("args", "") or ""
    if name == "TaskStop":
        return "停止后台任务", json.dumps(inp, ensure_ascii=False)
    if name == "ToolSearch":
        return "加载工具 " + inp.get("query", ""), ""
    return name, json.dumps(inp, ensure_ascii=False)


def _tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


def slash_command(text: str) -> dict | None:
    """Return a user/note event for a slash command, {} to drop it, or None if not a command."""
    if "<command-name>" not in text[:200] and "<command-message>" not in text[:200]:
        return None
    name, args = _tag(text, "command-name"), _tag(text, "command-args")
    if name == "/dforge":
        return dict(k="user", x=f"/dforge {args}".strip())
    if name == "/goal":
        first = args.split("\n", 1)[0].strip()
        return dict(k="note", x=f"设置 /goal 持续目标：{first}" if args else "设置 /goal 持续目标")
    return {}


def notification_text(text: str) -> str:
    summary = _tag(text, "summary")
    event = _tag(text, "event")
    if summary.startswith("Goal check-in"):
        return "Goal 检查：后台作业仍在运行，agent 查看进度后继续等待"
    if "didn't finish before the previous session ended" in summary or "didn\'t finish" in summary:
        return "后台通知：上一会话遗留的后台任务已标记停止"
    if event and not event.startswith("[Monitor expired"):
        return f"后台通知：{summary} → {event[:160]}"
    if event.startswith("[Monitor expired"):
        return f"后台通知：{summary}（30 分钟无新事件，监控到期）"
    return f"后台通知：{summary}"


def parse_claude(path: Path) -> list[dict]:
    events: list[dict] = []
    calls: dict[str, dict] = {}
    asks: dict[str, dict] = {}
    for line in path.open():
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get("type")
        if typ not in ("user", "assistant") or o.get("isSidechain"):
            continue
        ts = o.get("timestamp", "")
        content = (o.get("message") or {}).get("content")
        if typ == "user":
            parts = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
            for part in parts:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    slash = slash_command(text)
                    if slash is not None:
                        if slash:
                            events.append(dict(t=ts, **slash))
                        continue
                    if text.startswith("<task-notification>"):
                        events.append(dict(t=ts, k="note", x=notification_text(text)))
                        continue
                    if o.get("isMeta") or text.startswith("<local-command") or text.startswith("<system-reminder>"):
                        continue
                    if o.get("isCompactSummary") or text.startswith("This session is being continued"):
                        events.append(dict(t=ts, k="note", x="上下文已压缩（/compact），会话在摘要基础上继续"))
                        continue
                    if text.startswith("[Request interrupted"):
                        events.append(dict(t=ts, k="note", x="用户中断了当前操作"))
                        continue
                    if text.strip() in ("/compact",):
                        continue
                    events.append(dict(t=ts, k="user", x=text))
                elif part.get("type") == "tool_result":
                    cid = part.get("tool_use_id")
                    c = part.get("content")
                    text = c if isinstance(c, str) else "\n".join(x.get("text", "") for x in (c or []) if isinstance(x, dict) and x.get("type") == "text")
                    if cid in asks:
                        events.append(dict(t=ts, k="answer", answers=CLAUDE_ANSWER.findall(text)))
                        continue
                    ev = calls.get(cid)
                    if ev is not None:
                        ev["o"] = text
                        ev["e"] = bool(part.get("is_error"))
        else:
            for part in content or []:
                if part.get("type") == "text" and part.get("text", "").strip():
                    text = part["text"]
                    if text.startswith("API Error"):
                        events.append(dict(t=ts, k="note", x="推理网关临时返回 502，agent 自动重试后继续"))
                    elif text.strip() == "No response requested.":
                        continue
                    else:
                        events.append(dict(t=ts, k="assistant", x=text))
                elif part.get("type") == "tool_use":
                    name, inp = part.get("name"), part.get("input") or {}
                    if name == "AskUserQuestion":
                        ev = dict(t=ts, k="ask", questions=[
                            dict(q=q.get("question", ""), header=q.get("header", ""),
                                 options=[op.get("label", "") for op in q.get("options", [])],
                                 details=[op.get("description", "") for op in q.get("options", [])])
                            for q in inp.get("questions", [])])
                        asks[part.get("id")] = ev
                        events.append(ev)
                        continue
                    title, body = claude_tool(name, inp)
                    ev = dict(t=ts, k="tool", n=name, c=title, b=body, o="", e=False)
                    calls[part.get("id")] = ev
                    events.append(ev)
    return events


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
STATUS_HINT = re.compile(r"\d+\s*/\s*[\d,]+|已收到|已处理|进度|仍在|仍为|排队|正在|运行正常|没有错误|没有异常")


def is_status(text: str) -> bool:
    flat = text.strip()
    return len(flat) <= 120 and "\n" not in flat and bool(STATUS_HINT.search(flat)) and "|" not in flat


def merge_answers(events: list[dict]) -> list[dict]:
    """Fold answer events into the preceding ask event."""
    out: list[dict] = []
    pending: list[dict] = []
    for ev in events:
        if ev["k"] == "ask":
            ev["answers"] = []
            pending.append(ev)
            out.append(ev)
        elif ev["k"] == "answer":
            for q, a in ev["answers"]:
                target = None
                for ask in pending:
                    if any(_same(q, item["q"]) for item in ask["questions"]):
                        target = ask
                        break
                target = target or (pending[-1] if pending else None)
                if target is not None:
                    target["answers"].append(dict(q=q, a=a, t=ev["t"]))
        else:
            out.append(ev)
    return out


def _same(a: str, b: str) -> bool:
    return a.strip()[:40] == b.strip()[:40]


def node_for(ts: str, breakpoints: list[tuple[str, str, int]]) -> tuple[str, int]:
    node, rnd = breakpoints[0][1], breakpoints[0][2]
    for start, n, r in breakpoints:
        if ts_key(ts) >= start:
            node, rnd = n, r
        else:
            break
    return node, rnd



# --------------------------------------------------------------------------------------
# Editorial pass (see curation.py)
# --------------------------------------------------------------------------------------
def load_curation() -> dict:
    import importlib.util
    path = Path(__file__).with_name("curation.py")
    spec = importlib.util.spec_from_file_location("demo_curation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BAD = re.compile(r"(?i)\bissues?\b|(?-i:(?<![A-Za-z])PRs?(?![A-Za-z]))|pull request|github|Oscar-dzy|\bgh (?:issue|pr|api|repo|auth)\b|#2[0-9]\b")
ISSUE_PATH = re.compile(r"(?i)-?issue\d+-?")


def scrub_paths(text: str) -> str:
    return ISSUE_PATH.sub(lambda m: "-" if m.group(0).startswith("-") and m.group(0).endswith("-") else "", text or "")


def scrub_text(text: str) -> str:
    """Drop list/table lines and sentences that still talk about issues, PRs or GitHub."""
    out = []
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        if not BAD.search(line):
            out.append(line)
            continue
        stripped = line.strip()
        if in_code or re.match(r"^([-*]|\d+\.|\||#|Sources)", stripped):
            continue
        parts = re.split(r"(?<=[。；！？.!?])\s*", line)
        kept = "".join(x for x in parts if x and not BAD.search(x)).strip()
        if kept:
            out.append(kept)
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # headings left without content
    text = re.sub(r"(^|\n)#{1,4} [^\n]*\n*(?=(\n#{1,4} |\Z))", r"\1", text)
    return text.strip()


def clean(text: str, replace: list) -> str:
    text = scrub_paths(redact(text))
    for old, new in replace:
        text = text.replace(old, new)
    return text


def curate(events: list[dict], cur: dict, replace: list, sub_summary: dict) -> list[dict]:
    rewrite = dict(cur.get("rewrite", {}))
    used = {k: 0 for k in rewrite}
    edits = cur.get("edits", {})
    out = []
    for ev in events:
        t = ts_key(ev["t"])
        if t > cur["end"] or any(a <= t <= b for a, b in cur.get("drop_windows", [])):
            continue
        k = ev["k"]
        if k == "note":
            continue
        if k in ("user", "assistant"):
            key = next((key for key in rewrite if t.startswith(key)), None)
            if key is not None:
                used[key] += 1
                if rewrite[key] is None:
                    continue
                ev = dict(ev, x=rewrite[key])
            for key, pairs in edits.items():
                if t.startswith(key):
                    x = redact(ev["x"])
                    for old, new in pairs:
                        if old not in x:
                            print(f"  ! edit not applied at {t}: {old[:40]!r}")
                        x = x.replace(old, new)
                    ev = dict(ev, x=x)
            ev = dict(ev, x=clean(ev["x"], replace))
            if k == "assistant":
                ev["x"] = scrub_text(ev["x"])
                if not ev["x"]:
                    continue
        elif k == "session":
            ev = dict(ev, x=cur["sessions"].get(ev["s"], ev["x"]))
        elif k == "ask":
            qs = []
            for q in ev["questions"]:
                text = q["q"]
                for old, new in cur.get("questions", []):
                    text = text.replace(old, new)
                def qfix(x: str) -> str:
                    for o, n in cur.get("questions", []):
                        x = x.replace(o, n)
                    return clean(x, replace)
                qs.append(dict(q, q=clean(text, replace), header=qfix(q.get("header", "")),
                               options=[qfix(o) for o in q.get("options", [])],
                               details=[qfix(d) for d in q.get("details", [])]))
            ans = []
            for a in ev["answers"]:
                q = a["q"]
                for old, new in cur.get("questions", []):
                    q = q.replace(old, new)
                ans.append(dict(a, q=clean(q, replace), a=cur.get("answers", {}).get(a["a"], clean(a["a"], replace))))
            ev = dict(ev, questions=qs, answers=ans)
        elif k == "subagent":
            name, text = cur.get("subagents", {}).get(ev["name"], (ev["name"], sub_summary.get(ev["name"], "")))
            ev = dict(ev, label=name, x=text)
        elif k == "tool":
            c, b = clean(ev["c"], replace), clean(ev["b"], replace)
            if BAD.search(c) or BAD.search(b):
                continue
            o = "\n".join(line for line in clean(ev["o"], replace).split("\n") if not BAD.search(line))
            ev = dict(ev, c=c, b=b, o=o)
        out.append(ev)
    for key, n in used.items():
        if n != 1:
            print(f"  ! rewrite key {key} matched {n} events")
    return out


def build(export: Path, tid: str, spec: dict, curation) -> dict:
    group = export / spec["group"]
    events: list[dict] = []
    for idx, (fname, label) in enumerate(spec["sessions"]):
        parser = parse_codex if fname.startswith("rollout-") else parse_claude
        sess = merge_answers(parser(group / fname))
        if not sess:
            continue
        events.append(dict(t=sess[0]["t"], k="session", x=label, s=idx + 1))
        events.extend(sess)

    sub_summary = {name: codex_subagent_summary(group / f) for name, f in spec["subagents"].items()}

    # Re-sort by time, keeping the session marker first among equal timestamps.
    events.sort(key=lambda e: (ts_key(e["t"]), 0 if e["k"] == "session" else 1))
    events = curate(events, curation.CURATION[tid], curation.REPLACE, sub_summary)

    out: list[dict] = []
    group_ev: dict | None = None
    stats = dict(tools=0, user=0, assistant=0, asks=0, subagents=0, errors=0)
    tool_names: dict[str, int] = {}
    for ev in events:
        node, rnd = node_for(ev["t"], spec["breakpoints"])
        if ev["k"] == "tool":
            stats["tools"] += 1
            stats["errors"] += int(ev["e"])
            tool_names[ev["n"]] = tool_names.get(ev["n"], 0) + 1
            item = dict(n=ev["n"], c=redact(ev["c"]), b=clip(redact(ev["b"]), CMD_LIMIT),
                        o=clip(redact(ev["o"]), OUT_LIMIT), e=ev["e"], t=ev["t"])
            if group_ev is None or group_ev["n"] != node or group_ev["r"] != rnd or len(group_ev["items"]) >= GROUP_LIMIT:
                group_ev = dict(k="tools", t=ev["t"], n=node, r=rnd, items=[])
                out.append(group_ev)
            group_ev["items"].append(item)
            continue
        group_ev = None
        base = dict(k=ev["k"], t=ev["t"], n=node, r=rnd)
        if ev["k"] == "user":
            stats["user"] += 1
            base["x"] = redact(ev["x"])
        elif ev["k"] == "assistant":
            stats["assistant"] += 1
            base["x"] = redact(ev["x"])
            if is_status(ev["x"]):
                base["k"] = "status"
        elif ev["k"] == "ask":
            stats["asks"] += 1
            base["questions"] = [dict(q=redact(q["q"]), header=q.get("header", ""),
                                      options=[redact(o) for o in q.get("options", [])],
                                      details=[redact(d) for d in q.get("details", [])]) for q in ev["questions"]]
            base["answers"] = [dict(q=redact(a["q"]), a=redact(a["a"]), t=a["t"]) for a in ev["answers"]]
        elif ev["k"] == "subagent":
            stats["subagents"] += 1
            base["name"] = ev.get("label", ev["name"])
            base["model"] = ev.get("model", "")
            base["x"] = clip(redact(ev.get("x", "")), 1600)
        elif ev["k"] in ("note", "session"):
            base["x"] = redact(ev["x"])
            if ev["k"] == "session":
                base["s"] = ev["s"]
        out.append(base)

    first, last = events[0]["t"], events[-1]["t"]
    span = (_dt(last) - _dt(first)).total_seconds()
    active = _active_seconds([e["t"] for e in events])
    return dict(
        id=tid,
        harness=spec["harness"],
        start=first,
        end=last,
        stats=dict(stats, sessions=len(spec["sessions"]), spanHours=round(span / 3600, 1),
                   activeHours=round(active / 3600, 1), toolNames=tool_names, events=len(out)),
        events=out,
    )


def _dt(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def _active_seconds(stamps: list[str], gap: float = 1800) -> float:
    """Sum of time between consecutive events, ignoring idle gaps longer than ``gap``."""
    times = sorted(_dt(s) for s in stamps)
    total = 0.0
    for a, b in zip(times, times[1:]):
        d = (b - a).total_seconds()
        if d <= gap:
            total += d
    return total


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    export = Path(sys.argv[1]).resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    curation = load_curation()
    for tid, spec in TRAJECTORIES.items():
        data = build(export, tid, spec, curation)
        def migrate(value):
            if isinstance(value, str):
                value = redact(value)
                for old, new in curation.COMMAND_REPLACEMENTS:
                    value = value.replace(old, new)
                return value
            if isinstance(value, list):
                return [migrate(item) for item in value]
            if isinstance(value, dict):
                return {key: migrate(item) for key, item in value.items()}
            return value
        data = migrate(data)
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        target = OUT_DIR / f"trajectory-{tid}.js"
        target.write_text(
            "// Generated by demo/scripts/build_trajectories.py — do not edit by hand.\n"
            "window.DFORGE_TRAJECTORIES = window.DFORGE_TRAJECTORIES || {};\n"
            f"window.DFORGE_TRAJECTORIES[{json.dumps(tid)}] = {payload};\n",
            encoding="utf-8",
        )
        s = data["stats"]
        print(f"{tid}: {s['events']} events · {s['tools']} tool calls · {s['user']} user msgs · "
              f"{s['asks']} asks · {s['subagents']} subagents · {target.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
