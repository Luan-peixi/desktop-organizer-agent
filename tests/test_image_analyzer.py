"""Tests for bounded, opt-in OpenAI image understanding."""

from pathlib import Path
from types import SimpleNamespace

import desktop_agent.image_analyzer as analyzer_module
from desktop_agent.image_analyzer import (
    DEFAULT_VISION_MODEL,
    OpenAIImageAnalyzer,
    VISION_PROMPT,
)
from desktop_agent.models import ContentExtractionStatus
from desktop_agent.plan_validator import index_files
from desktop_agent.scanner import scan_directory


class FakeResponsesAPI:
    """Record vision requests and return a configured description."""

    def __init__(self, output_text: str = "线性代数课件，包含矩阵公式") -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class FakeClient:
    """Minimal OpenAI-compatible client for vision tests."""

    def __init__(self, output_text: str = "线性代数课件，包含矩阵公式") -> None:
        self.responses = FakeResponsesAPI(output_text)


def _scan_one(directory: Path):
    files = scan_directory(directory)
    assert len(files) == 1
    return files[0]


def test_image_is_sent_as_data_url_without_local_path(tmp_path: Path) -> None:
    """The API should receive bytes and the defensive prompt, never a path."""

    path = tmp_path / "课堂截图.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
    client = FakeClient()

    result = OpenAIImageAnalyzer(client=client).analyze_image(
        _scan_one(tmp_path)
    )

    assert result.content_status is ContentExtractionStatus.EXTRACTED
    assert result.content_excerpt == "线性代数课件，包含矩阵公式"
    request = client.responses.calls[0]
    assert request["model"] == DEFAULT_VISION_MODEL
    assert request["reasoning"] == {"effort": "low"}
    assert request["store"] is False
    request_text = str(request)
    assert "data:image/jpeg;base64," in request_text
    assert str(tmp_path) not in request_text
    request_input = request["input"]
    assert isinstance(request_input, list)
    content = request_input[0]["content"]
    assert content[0]["text"] == VISION_PROMPT


def test_oversized_image_is_not_uploaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A file above the local limit should not produce an API request."""

    path = tmp_path / "large.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n12345")
    monkeypatch.setattr(analyzer_module, "MAX_IMAGE_BYTES", 4)
    client = FakeClient()

    result = OpenAIImageAnalyzer(client=client).analyze_image(
        _scan_one(tmp_path)
    )

    assert result.content_status is ContentExtractionStatus.TOO_LARGE
    assert client.responses.calls == []


def test_empty_vision_response_degrades_to_error(tmp_path: Path) -> None:
    """An unusable response should preserve the file for metadata fallback."""

    path = tmp_path / "unknown.webp"
    path.write_bytes(b"RIFF1234WEBPfake")

    result = OpenAIImageAnalyzer(client=FakeClient(" ")).analyze_image(
        _scan_one(tmp_path)
    )

    assert result.content_status is ContentExtractionStatus.ERROR
    assert result.content_excerpt is None


def test_batch_limit_leaves_extra_images_unuploaded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """One analysis run should respect its image-count cost guardrail."""

    for index in range(3):
        (tmp_path / f"image-{index}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )
    monkeypatch.setattr(analyzer_module, "MAX_IMAGES_PER_RUN", 2)
    client = FakeClient()
    files = index_files(scan_directory(tmp_path))

    results = OpenAIImageAnalyzer(client=client).analyze_images(files)

    assert len(client.responses.calls) == 2
    assert sum(
        item.content_status is ContentExtractionStatus.EXTRACTED
        for item in results.values()
    ) == 2


def test_fake_image_extension_is_rejected_without_upload(tmp_path: Path) -> None:
    """Text renamed to .jpg must not be sent to the vision endpoint."""

    path = tmp_path / "not-an-image.jpg"
    path.write_text("plain text")
    client = FakeClient()

    result = OpenAIImageAnalyzer(client=client).analyze_image(
        _scan_one(tmp_path)
    )

    assert result.content_status is ContentExtractionStatus.ERROR
    assert client.responses.calls == []
