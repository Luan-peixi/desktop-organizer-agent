"""Tests for read-only directory scanning."""

from datetime import UTC
from pathlib import Path

import pytest

import desktop_agent.scanner as scanner_module
from desktop_agent.scanner import list_selectable_entries, scan_directory


def test_scan_directory_returns_sorted_metadata(tmp_path: Path) -> None:
    """Visible files should be returned with normalized metadata."""

    (tmp_path / "zeta.PDF").write_bytes(b"123")
    (tmp_path / "alpha.txt").write_bytes(b"hello")

    files = scan_directory(tmp_path)

    assert [file.name for file in files] == ["alpha.txt", "zeta.PDF"]
    assert [file.extension for file in files] == [".txt", ".pdf"]
    assert [file.size_bytes for file in files] == [5, 3]
    assert all(file.path.parent == tmp_path for file in files)
    assert all(file.modified_at.tzinfo is UTC for file in files)


def test_scan_directory_includes_safe_container_and_ignores_unsafe_entries(
    tmp_path: Path,
) -> None:
    """Visible folders are containers; hidden entries and links stay ignored."""

    (tmp_path / "visible.txt").write_bytes(b"visible")
    (tmp_path / ".hidden.txt").write_bytes(b"hidden")
    (tmp_path / ".desktop_agent_last_execution.json").write_text("{}")

    nested_directory = tmp_path / "nested"
    nested_directory.mkdir()
    nested_file = nested_directory / "inside.txt"
    nested_file.write_bytes(b"nested")

    (tmp_path / "linked.txt").symlink_to(nested_file)

    files = scan_directory(tmp_path)

    assert [file.name for file in files] == ["nested", "visible.txt"]
    assert files[0].is_directory is True
    assert files[0].tree_fingerprint is not None
    assert files[1].is_directory is False


def test_scan_directory_rejects_missing_path(tmp_path: Path) -> None:
    """A missing scan target should raise FileNotFoundError."""

    missing_directory = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_directory(missing_directory)


def test_scan_directory_rejects_file_path(tmp_path: Path) -> None:
    """A regular file cannot be used as the scan target."""

    file_path = tmp_path / "not-a-directory.txt"
    file_path.write_bytes(b"content")

    with pytest.raises(NotADirectoryError):
        scan_directory(file_path)


def test_scan_directory_does_not_modify_source_files(tmp_path: Path) -> None:
    """Scanning should not change directory entries or file data."""

    file_path = tmp_path / "important.txt"
    original_content = b"do not change"
    file_path.write_bytes(original_content)

    entries_before = sorted(path.name for path in tmp_path.iterdir())
    file_stat_before = file_path.stat()

    scan_directory(tmp_path)

    entries_after = sorted(path.name for path in tmp_path.iterdir())
    file_stat_after = file_path.stat()

    assert entries_after == entries_before
    assert file_path.read_bytes() == original_content
    assert file_stat_after.st_size == file_stat_before.st_size
    assert file_stat_after.st_mtime_ns == file_stat_before.st_mtime_ns


def test_scan_directory_ignores_agent_category_directories(
    tmp_path: Path,
) -> None:
    """A prior output folder must not become a new container input."""

    category = tmp_path / "学习资料"
    category.mkdir()
    (category / ".desktop_agent_category").write_text("marker")
    (category / "课程作业.txt").write_text("content")
    (tmp_path / "new.txt").write_text("new")

    files = scan_directory(tmp_path)

    assert [file.name for file in files] == ["new.txt"]


def test_scan_directory_only_indexes_selected_top_level_entries(
    tmp_path: Path,
) -> None:
    """Unselected desktop items must remain outside the entire workflow."""

    (tmp_path / "selected.txt").write_text("selected")
    selected_folder = tmp_path / "selected-folder"
    selected_folder.mkdir()
    (selected_folder / "report.txt").write_text("report")
    (tmp_path / "leave-alone.txt").write_text("private")

    files = scan_directory(
        tmp_path,
        selected_names={"selected.txt", "selected-folder"},
    )

    assert [file.name for file in files] == [
        "selected-folder",
        "selected.txt",
    ]


def test_running_agent_project_is_hidden_and_cannot_be_force_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The project containing the running source must never enter a scan."""

    project = tmp_path / "renamed-agent-project"
    project.mkdir()
    (project / "app.py").write_text("application")
    (tmp_path / "notes.txt").write_text("notes")
    monkeypatch.setattr(scanner_module, "AGENT_PROJECT_ROOT", project)

    assert list_selectable_entries(tmp_path) == ["notes.txt"]

    files = scan_directory(
        tmp_path,
        selected_names={"renamed-agent-project", "notes.txt"},
    )

    assert [file.name for file in files] == ["notes.txt"]


def test_parent_folder_containing_agent_project_is_also_protected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving a parent container must not indirectly move the Agent."""

    parent = tmp_path / "development"
    project = parent / "desktop-agent"
    project.mkdir(parents=True)
    monkeypatch.setattr(scanner_module, "AGENT_PROJECT_ROOT", project)

    assert list_selectable_entries(tmp_path) == []
    assert scan_directory(tmp_path) == []
