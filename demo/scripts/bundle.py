#!/usr/bin/env python3
"""Bundle the demo into one self-contained HTML file for sharing.

Usage:
    python demo/scripts/bundle.py [output.html]

Inlines styles, scripts, trajectory data and the logo, so the result opens
offline with a double click. Default output: demo/dist/decisionforge-demo.html
"""
import base64
import re
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEMO / "dist" / "decisionforge-demo.html"
    html = (DEMO / "index.html").read_text(encoding="utf-8")

    def css(m: re.Match) -> str:
        return "<style>\n" + (DEMO / m.group(1)).read_text(encoding="utf-8") + "\n</style>"

    def js(m: re.Match) -> str:
        code = (DEMO / m.group(1)).read_text(encoding="utf-8").replace("</script", "<\\/script")
        return "<script>\n" + code + "\n</script>"

    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', css, html)
    html = re.sub(r'<script src="([^"]+)"></script>', js, html)
    def inline_asset(match):
        relative = match.group(2)
        path = DEMO / relative
        if not path.is_file():
            return match.group(0)
        mime = {".svg": "image/svg+xml", ".png": "image/png"}.get(path.suffix)
        if not mime:
            return match.group(0)
        data = base64.b64encode(path.read_bytes()).decode()
        return f'{match.group(1)}="data:{mime};base64,{data}"'

    html = re.sub(r'(src|href)="(assets/[^"<>]+)"', inline_asset, html)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
