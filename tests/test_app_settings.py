"""Tests for safe Streamlit analysis settings."""

from pathlib import Path

from app import _selection_contains_containers


def test_directory_selection_requires_content_reading(tmp_path: Path) -> None:
    """Folder classification should not silently fall back to its outer name."""

    container = tmp_path / "20260001_示例学生_实验5"
    container.mkdir()

    assert _selection_contains_containers(str(tmp_path), [container.name])


def test_zip_selection_requires_content_reading(tmp_path: Path) -> None:
    """ZIP metadata alone is insufficient for reliable organization."""

    archive = tmp_path / "experiment.zip"
    archive.write_bytes(b"not needed by this setting test")

    assert _selection_contains_containers(str(tmp_path), [archive.name])


def test_regular_file_can_still_use_metadata_only_mode(tmp_path: Path) -> None:
    """A user may deliberately classify ordinary files without body content."""

    document = tmp_path / "人工智能课件.pdf"
    document.write_bytes(b"metadata only")

    assert not _selection_contains_containers(str(tmp_path), [document.name])
