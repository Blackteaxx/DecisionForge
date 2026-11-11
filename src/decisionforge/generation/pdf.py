"""MinerU v4 batch upload -> durable UTF-8 TXT corpus -> existing generation intake."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
from filelock import FileLock

from decisionforge.util import PipelineError, atomic_write, digest, write_json

logger = logging.getLogger(__name__)
MAX_PDF_BYTES = 200 * 1024 * 1024
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_TEXT_BYTES = 64 * 1024 * 1024


class MinerUError(PipelineError):
    pass


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pdf_files(input_config):
    """Resolve explicit files or a directory, never silently include unrelated formats."""
    if input_config.paths and input_config.path:
        raise ValueError("PDF input: use either input.path or input.paths, not both")
    paths = [Path(p).resolve() for p in input_config.paths]
    if not paths:
        if not input_config.path:
            raise ValueError("PDF input requires input.path or input.paths")
        root = Path(input_config.path).resolve()
        paths = (
            sorted(p for p in root.iterdir() if p.suffix.lower() == ".pdf")
            if root.is_dir()
            else [root]
        )
    if not paths:
        raise ValueError("No PDF files found in input directory")
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate PDF input paths")
    for path in paths:
        if path.suffix.lower() != ".pdf" or not path.is_file():
            raise ValueError(f"PDF input is not a PDF file: {path}")
        size = path.stat().st_size
        if not 0 < size <= MAX_PDF_BYTES:
            raise ValueError(f"PDF must be nonempty and no larger than 200 MiB: {path.name}")
        with path.open("rb") as source:
            if b"%PDF-" not in source.read(1024):
                raise ValueError(f"Invalid PDF header: {path.name}")
    return paths


def inspect_pdf(config):
    paths = pdf_files(config.input)
    return {
        "type": "pdf",
        "records": len(paths),
        "files": [{"path": str(p), "bytes": p.stat().st_size} for p in paths],
        "parser": "MinerU",
        "model_version": config.mineru.model_version,
        "credential_environment": config.mineru.api_key_env,
        "note": "Offline file validation only; build uploads PDFs to MinerU and creates TXT. "
        "MinerU enforces its page limit (200 pages per file).",
    }


class MinerUClient:
    def __init__(self, settings, client):
        self.settings, self.client = settings, client
        self.key = os.getenv(settings.api_key_env)
        if not self.key:
            raise MinerUError(
                f"Set {settings.api_key_env} in environment or provide MinerU API Key"
            )

    async def api(self, method, route, **kwargs):
        response = await self.client.request(
            method,
            self.settings.base_url.rstrip("/") + route,
            headers={"Authorization": f"Bearer {self.key}"},
            **kwargs,
        )
        response.raise_for_status()
        try:
            result = response.json()
        except ValueError:
            raise MinerUError("MinerU returned invalid JSON") from None
        if not isinstance(result, dict) or result.get("code") != 0:
            # Do not echo an arbitrary remote body, which can contain credentials or signed URLs.
            code = result.get("code") if isinstance(result, dict) else None
            raise MinerUError(
                f"MinerU API rejected request (code={code if isinstance(code, int) else 'unknown'})"
            )
        if not isinstance(result.get("data"), dict):
            raise MinerUError("MinerU response is missing data")
        return result["data"]

    async def submit(self, paths, identifiers):
        data = await self.api(
            "POST",
            "/file-urls/batch",
            json={
                "files": [
                    {"name": p.name, "data_id": identifier, "is_ocr": self.settings.is_ocr}
                    for p, identifier in zip(paths, identifiers, strict=True)
                ],
                "model_version": self.settings.model_version,
                "language": self.settings.language,
                "enable_formula": self.settings.enable_formula,
                "enable_table": self.settings.enable_table,
            },
        )
        batch_id, urls = data.get("batch_id"), data.get("file_urls")
        if not isinstance(batch_id, str) or not re.fullmatch(r"[\w-]+", batch_id):
            raise MinerUError("MinerU returned an invalid batch ID")
        if not isinstance(urls, list) or len(urls) != len(paths):
            raise MinerUError("MinerU upload URL count does not match the PDF count")
        for path, url in zip(paths, urls, strict=True):
            logger.info("[PDF] Uploading %s", path.name)

            async def chunks():
                with path.open("rb") as source:
                    while block := source.read(1024 * 1024):
                        yield block

            # Signed object-storage URLs must never receive the MinerU bearer token.
            response = await self.client.put(
                url,
                content=chunks(),
                headers={
                    "Content-Length": str(path.stat().st_size),
                },
            )
            response.raise_for_status()
        return batch_id

    async def markdown(self, url):
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as archive:
            async with self.client.stream("GET", url, follow_redirects=True) as response:
                response.raise_for_status()
                size = 0
                async for block in response.aiter_bytes():
                    size += len(block)
                    if size > MAX_ARCHIVE_BYTES:
                        raise MinerUError("MinerU result archive exceeds 512 MiB")
                    archive.write(block)
            archive.seek(0)
            try:
                with zipfile.ZipFile(archive) as bundle:
                    # Read a member directly: never extract remote paths onto the filesystem.
                    entries = [
                        i
                        for i in bundle.infolist()
                        if not i.is_dir() and i.filename.lower().endswith(".md")
                    ]
                    preferred = [i for i in entries if Path(i.filename).name == "full.md"]
                    if len(preferred) == 1:
                        entry = preferred[0]
                    elif len(entries) == 1:
                        entry = entries[0]
                    else:
                        raise MinerUError("MinerU archive has missing or ambiguous Markdown output")
                    if entry.file_size > MAX_TEXT_BYTES:
                        raise MinerUError("MinerU Markdown exceeds 64 MiB")
                    text = bundle.read(entry).decode("utf-8-sig")
            except (zipfile.BadZipFile, UnicodeError, RuntimeError):
                raise MinerUError("MinerU result is not a valid UTF-8 Markdown ZIP") from None
        if not text.strip():
            raise MinerUError("MinerU returned empty document text")
        return text

    async def collect(self, batch_id, entries, directory, checkpoint):
        async with asyncio.timeout(self.settings.max_wait_seconds):
            while any(not entry.get("text_sha256") for entry in entries):
                data = await self.api("GET", f"/extract-results/batch/{batch_id}")
                results = data.get("extract_result")
                if not isinstance(results, list):
                    raise MinerUError("MinerU result is missing extract_result")
                by_id = {item.get("data_id"): item for item in results if isinstance(item, dict)}
                for entry in entries:
                    if entry.get("text_sha256"):
                        continue
                    item = by_id.get(entry["data_id"])
                    if item is None:
                        continue
                    if item.get("state") == "failed":
                        raise MinerUError(
                            f"MinerU parsing failed for {Path(entry['source']).name}; batch={batch_id}"
                        )
                    if item.get("state") != "done":
                        continue
                    url = item.get("full_zip_url")
                    if not isinstance(url, str) or not url:
                        raise MinerUError("MinerU completed without a result archive URL")
                    text = await self.markdown(url)
                    path = directory / entry["txt"]
                    atomic_write(path, text)
                    entry["text_sha256"] = sha256_file(path)
                    write_json(checkpoint, {"batch_id": batch_id, "entries": entries})
                    logger.info("[PDF] Parsed %s -> %s", Path(entry["source"]).name, path)
                if any(not entry.get("text_sha256") for entry in entries):
                    await asyncio.sleep(self.settings.poll_interval_seconds)


async def prepare_pdf(config, *, client=None, output_root=None):
    """Return a cached corpus path. Failed/partial parses never enter JEV generation."""
    paths = pdf_files(config.input)
    sources = [{"source": str(p), "sha256": sha256_file(p)} for p in paths]
    settings = config.mineru.model_dump(
        exclude={
            "api_key_env",
            "timeout_seconds",
            "poll_interval_seconds",
            "max_wait_seconds",
        }
    )
    identity = digest({"version": 1, "sources": sources, "settings": settings})
    root = (
        Path(output_root) if output_root else Path(config.output.directory).parent / ".dforge" / "pdf"
    )
    directory = root.resolve() / identity
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "corpus.txt"
    manifest_path = directory / "manifest.json"
    started = time.monotonic()
    with FileLock(str(directory / ".parse.lock"), timeout=0):
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if corpus.is_file() and sha256_file(corpus) == saved["corpus_sha256"]:
                return corpus
            raise MinerUError("Cached PDF corpus was modified; use a new parse output directory")
        own_client = client is None
        client = client or httpx.AsyncClient(timeout=config.mineru.timeout_seconds)
        try:
            mineru = MinerUClient(config.mineru, client)
            all_entries = []
            for start in range(0, len(paths), config.mineru.batch_size):
                batch_paths = paths[start : start + config.mineru.batch_size]
                entries = [
                    {**source, "data_id": f"pdf_{i:05d}", "txt": f"{i:05d}.txt"}
                    for i, source in enumerate(sources[start : start + len(batch_paths)], start)
                ]
                checkpoint = directory / f"batch_{start:05d}.json"
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                    batch_id, entries = saved["batch_id"], saved["entries"]
                    for entry in entries:
                        if entry.get("text_sha256") and (
                            not (directory / entry["txt"]).exists()
                            or sha256_file(directory / entry["txt"]) != entry["text_sha256"]
                        ):
                            raise MinerUError(
                                "Cached PDF text was modified; use a new parse output directory"
                            )
                else:
                    batch_id = await mineru.submit(
                        batch_paths, [entry["data_id"] for entry in entries]
                    )
                    write_json(checkpoint, {"batch_id": batch_id, "entries": entries})
                await mineru.collect(batch_id, entries, directory, checkpoint)
                all_entries.extend(entries)
            text = "\n\n".join(
                f"===== PDF: {Path(entry['source']).name} =====\n\n"
                + (directory / entry["txt"]).read_text(encoding="utf-8")
                for entry in all_entries
            )
            atomic_write(corpus, text + "\n")
            write_json(
                manifest_path,
                {
                    "parser": "MinerU",
                    "documentation": "https://mineru.net/apiManage/docs",
                    "settings": settings,
                    "files": all_entries,
                    "corpus": str(corpus),
                    "corpus_sha256": sha256_file(corpus),
                },
            )
            logger.info(
                "[PDF] Corpus ready (%s PDFs, %.1fs): %s",
                len(paths),
                time.monotonic() - started,
                corpus,
            )
            return corpus
        except TimeoutError:
            raise MinerUError("MinerU parsing timed out; rerun to resume the saved batch") from None
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            raise MinerUError(
                f"MinerU transport failed ({type(exc).__name__}, HTTP {status}); rerun to retry"
            ) from None
        finally:
            if own_client:
                await client.aclose()
