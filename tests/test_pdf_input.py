"""MinerU batch protocol and PDF intake regression coverage; never calls the live API."""

import io
import json
import zipfile

import httpx
import pytest

from decisionforge.cli import execute, inspect_input, main
from decisionforge.config import Config, load_config
from decisionforge.generation.pdf import MinerUError, pdf_files, prepare_pdf


def pdf_config(tmp_path, count=2):
    folder = tmp_path / "pdf"
    folder.mkdir()
    for i in range(count):
        (folder / f"paper{i}.pdf").write_bytes(b"%PDF-1.7\nfixture")
    return Config(
        input={"type": "pdf", "path": str(folder)},
        output={"directory": str(tmp_path / "run")},
        mineru={"poll_interval_seconds": 0.001, "max_wait_seconds": 1},
    )


def archive(text, name="nested/full.md"):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        bundle.writestr(name, text)
        bundle.writestr("../should-not-extract", "unsafe path")
    return output.getvalue()


class MinerUFixture:
    def __init__(self):
        self.requests = []
        self.batches = {}
        self.state = "done"
        self.result = archive("# Parsed paper\nVoltage is 220 V.\n|A|B|\n|1|2|")

    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path
        if "/api/v4/" in path:
            assert request.headers["authorization"] == "Bearer test-mineru"
        else:
            assert "authorization" not in request.headers
        if path.endswith("/file-urls/batch"):
            body = json.loads(request.content)
            assert body["model_version"] == "vlm"
            batch = str(len(self.batches))
            self.batches[batch] = body["files"]
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "batch_id": batch,
                        "file_urls": [
                            f"https://storage.invalid/upload/{f['data_id']}" for f in body["files"]
                        ],
                    },
                },
            )
        if path.startswith("/upload/"):
            assert request.method == "PUT" and request.content.startswith(b"%PDF-")
            return httpx.Response(200)
        if "/extract-results/batch/" in path:
            batch = path.rsplit("/", 1)[-1]
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {
                        "extract_result": [
                            {
                                "data_id": file["data_id"],
                                "state": self.state,
                                "full_zip_url": "https://storage.invalid/result.zip",
                            }
                            for file in reversed(self.batches[batch])
                        ]
                    },
                },
            )
        if path == "/result.zip":
            return httpx.Response(200, content=self.result)
        raise AssertionError(str(request.url))


async def test_batch_parse_cache_manifest_and_no_extraction(tmp_path, monkeypatch):
    config = pdf_config(tmp_path, 3)
    config.mineru.batch_size = 2
    monkeypatch.setenv("MINERU_API_KEY", "test-mineru")
    fixture = MinerUFixture()
    async with httpx.AsyncClient(transport=httpx.MockTransport(fixture)) as client:
        path = await prepare_pdf(config, client=client)
        assert path.read_text().count("Voltage is 220 V.") == 3
        assert path.read_text().index("paper0.pdf") < path.read_text().index("paper2.pdf")
        manifest = json.loads((path.parent / "manifest.json").read_text())
        assert len(manifest["files"]) == 3
        assert len(fixture.batches) == 2
        assert not (path.parent / "should-not-extract").exists()
        assert "test-mineru" not in (path.parent / "manifest.json").read_text()
        calls = len(fixture.requests)
        monkeypatch.delenv("MINERU_API_KEY")
        assert await prepare_pdf(config, client=client) == path
        assert len(fixture.requests) == calls
        path.write_text("modified")
        with pytest.raises(MinerUError, match="modified"):
            await prepare_pdf(config, client=client)


async def test_timeout_resumes_saved_batch_without_reupload(tmp_path, monkeypatch):
    config = pdf_config(tmp_path)
    config.mineru.max_wait_seconds = 0.01
    monkeypatch.setenv("MINERU_API_KEY", "test-mineru")
    fixture = MinerUFixture()
    fixture.state = "running"
    async with httpx.AsyncClient(transport=httpx.MockTransport(fixture)) as client:
        with pytest.raises(MinerUError, match="timed out"):
            await prepare_pdf(config, client=client)
        fixture.state = "done"
        config.mineru.max_wait_seconds = 1
        path = await prepare_pdf(config, client=client)
    assert path.exists() and len(fixture.batches) == 1
    assert len([r for r in fixture.requests if r.method == "PUT"]) == 2


@pytest.mark.parametrize(
    "failure", ["failed", "bad_zip", "empty", "no_markdown", "api_error", "missing_url"]
)
async def test_parse_errors_never_publish_corpus(tmp_path, monkeypatch, failure):
    config = pdf_config(tmp_path, 1)
    monkeypatch.setenv("MINERU_API_KEY", "test-mineru")
    fixture = MinerUFixture()
    if failure == "failed":
        fixture.state = "failed"
    elif failure == "bad_zip":
        fixture.result = b"not a zip"
    elif failure == "empty":
        fixture.result = archive(" ")
    elif failure == "no_markdown":
        fixture.result = archive("data", "result.json")

    def handler(request):
        if failure == "api_error":
            return httpx.Response(200, json={"code": -1, "msg": "test-mineru"})
        response = fixture(request)
        if failure == "missing_url" and "extract-results" in request.url.path:
            body = json.loads(response.content)
            body["data"]["extract_result"][0].pop("full_zip_url")
            return httpx.Response(200, json=body)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MinerUError) as exc:
            await prepare_pdf(config, client=client)
        assert "test-mineru" not in str(exc.value)
    assert not list(tmp_path.rglob("corpus.txt"))


def test_offline_inspection_paths_and_validation(tmp_path):
    config = pdf_config(tmp_path)
    assert inspect_input(config)["records"] == 2
    config.input.paths = [str(p) for p in pdf_files(config.input)]
    config.input.path = ""
    assert inspect_input(config)["type"] == "pdf"
    resolved = Config.model_validate(config.model_dump())
    assert len(resolved.input.paths) == 2
    config.input.paths.append(config.input.paths[0])
    with pytest.raises(ValueError, match="Duplicate"):
        inspect_input(config)


def test_relative_pdf_paths_and_auto_directory(tmp_path):
    config = pdf_config(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("input:\n  paths: [pdf/paper0.pdf, pdf/paper1.pdf]\n")
    loaded = load_config(path)
    assert loaded.input.type == "pdf"
    assert loaded.input.paths[0] == str(tmp_path / "pdf/paper0.pdf")
    assert inspect_input(config)["records"] == 2
    path.write_text("input: {path: pdf}\n")
    assert load_config(path).input.type == "pdf"


async def test_execute_pdf_uses_existing_txt_pipeline_without_mutating_config(
    tmp_path, monkeypatch
):
    config = pdf_config(tmp_path, 1)
    corpus = tmp_path / "parsed.txt"
    corpus.write_text("Existing text pipeline")
    seen = []

    async def prepare(value):
        assert value.input.type == "pdf"
        return corpus

    async def run(self):
        seen.append(self.config)
        assert self.config.input.type == "txt"
        assert self.config.input.path == str(corpus) and self.config.input.paths == []
        return {"status": "complete"}

    monkeypatch.setattr("decisionforge.generation.pdf.prepare_pdf", prepare)
    monkeypatch.setattr("decisionforge.pipeline.Pipeline.run", run)
    result = await execute(config)
    assert result["status"] == "complete" and len(seen) == 1
    assert config.input.type == "pdf"


def test_parse_command_does_not_call_llm(tmp_path, monkeypatch, capsys):
    pdf_config(tmp_path, 1)
    path = tmp_path / "config.yaml"
    path.write_text("input: {path: pdf}\n")

    async def prepare(config, **kwargs):
        assert config.input.type == "pdf"
        return tmp_path / "corpus.txt"

    monkeypatch.setattr("decisionforge.generation.pdf.prepare_pdf", prepare)
    assert main(["data", "parse-pdf", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["corpus"].endswith("corpus.txt")
