# DecisionForge brand assets

SVG sources are the canonical files. Rebuild the PNG exports from the repository root with:

```bash
uv run --with cairosvg python docs/assets/brand/export.py
```

CairoSVG also requires the native Cairo library: `brew install cairo` on macOS or `libcairo2` on Debian/Ubuntu.

`logo.svg` is the full-color app icon. `logo-mono.svg` follows `currentColor`. The two wordmarks provide transparent light- and dark-background variants. PNG exports are 32px favicon, 180px Apple touch icon, 512px plugin icon, and 1280x640 social preview. CairoSVG is an export-only tool and is not a project runtime dependency.

Use Inter 600 for the wordmark when available; SVG text falls back to the system sans-serif. Shared CSS tokens:

| Token | Light | Dark | Use |
| --- | --- | --- | --- |
| `--df-primary` | `#4F46E5` | `#A5B4FC` | Links and emphasis |
| `--df-primary-2` | `#7C3AED` | `#C4B5FD` | Gradient endpoint |
| `--df-ember` | `#F97316` | `#FB923C` | Selected or completed state |
| `--df-ink` | `#18181B` | `#F4F4F5` | Body text |
| `--df-muted` | `#6B6B76` | `#A1A1AA` | Secondary text |
| `--df-bg` | `#FAFAF9` | `#18171C` | Background |
| `--df-border` | `#E7E5E4` | `#3A3842` | Borders |
