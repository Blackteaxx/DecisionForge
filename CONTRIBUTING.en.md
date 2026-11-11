# Contributing

[中文](CONTRIBUTING.md)

Describe reproducible problems or proposals in an issue; small fixes may go directly to a PR. Read the relevant [development guide](docs/en/extending.md).

```bash
uv sync --locked --group dev
uv run ruff check
uv run pytest
uv run python tests/acceptance/check_public.py
```

Plugin changes also require Node >=24.9 and npm:

```bash
npm ci --prefix extensions/dsh-decisionforge
npm run build --prefix extensions/dsh-decisionforge
npm test --prefix extensions/dsh-decisionforge
```

Add meaningful regression coverage for behavioral changes. Preserve JEV semantics, source isolation, checkpoint recovery and credential boundaries for offline commands. Import optional dependencies lazily. GPU tests and paid model calls need agreed resources and budgets; offline success is not GPU acceptance.

Describe the problem, resulting behavior, validation and limitations in the PR. Keep Chinese and English guides aligned. Generate brand assets from `docs/assets/brand/`. Do not commit credentials, private addresses, weights, full personal sessions or local paths. Use synthetic or clearly licensed minimal fixtures, preserving third-party provenance and licenses.

Contributions use [Apache-2.0](LICENSE). Third-party code, models and datasets retain their licenses. For a leaked credential, revoke/rotate it first and contact the maintainer privately; never post the value in a public issue.
