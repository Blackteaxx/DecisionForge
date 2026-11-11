from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parent

for source, target, size in (
    ("favicon.svg", "favicon-32.png", 32),
    ("logo.svg", "icon-512.png", 512),
    ("favicon.svg", "apple-touch-icon.png", 180),
):
    cairosvg.svg2png(
        url=str(ROOT / source),
        write_to=str(ROOT / target),
        output_width=size,
        output_height=size,
    )

cairosvg.svg2png(
    url=str(ROOT / "social-preview.svg"),
    write_to=str(ROOT / "social-preview.png"),
    output_width=1280,
    output_height=640,
)
