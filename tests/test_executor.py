"""Tests for safe, read-only execution previews."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

import desktop_agent.executor as executor_module
import desktop_agent.scanner as scanner_module
from desktop_agent.executor import (
    ExecutionError,
    ExecutionPreview,
    ExecutionPreviewError,
    MoveOperation,
    UNDO_JOURNAL_NAME,
    build_execution_preview,
    execute_preview,
    get_last_execution_move_count,
    undo_last_execution,
)
from desktop_agent.models import (
    AssignmentStatus,
    CategoryDefinition,
    FileAssignment,
    FileMetadata,
    OrganizationPlan,
)
from desktop_agent.scanner import scan_directory


def _metadata(path: Path) -> FileMetadata:
    """Create metadata for a test file."""

    return FileMetadata(
        name=path.name,
        path=path,
        extension=path.suffix.lower(),
        size_bytes=path.stat().st_size,
        created_at=None,
        modified_at=datetime.now(UTC),
    )


def _plan(
    *,
    status: AssignmentStatus = AssignmentStatus.CLASSIFIED,
    category_name: str = "项目资料",
) -> OrganizationPlan:
    """Create a minimal valid plan for one file."""

    return OrganizationPlan(
        overview="将项目文件整理到项目资料目录。",
        categories=[
            CategoryDefinition(
                id="work",
                name="工作",
                description="工作相关文件。",
                parent_id=None,
            ),
            CategoryDefinition(
                id="projects",
                name=category_name,
                description="项目相关文件。",
                parent_id="work",
            ),
        ],
        assignments=[
            FileAssignment(
                file_id="file_001",
                category_id="projects",
                confidence=0.95,
                reason="文件名表明它是项目材料。",
                status=status,
            )
        ],
    )


def test_preview_builds_nested_move_without_touching_files(
    tmp_path: Path,
) -> None:
    """A preview should calculate a nested path without creating it."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")

    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )

    assert len(preview.operations) == 1
    assert preview.operations[0].source == source
    assert preview.operations[0].destination == (
        tmp_path / "工作" / "项目资料" / source.name
    )
    assert preview.skipped == ()
    assert source.exists()
    assert not (tmp_path / "工作").exists()


def test_preview_skips_files_that_need_review(tmp_path: Path) -> None:
    """A low-confidence file should never become an automatic move."""

    source = tmp_path / "无扩展名文件"
    source.write_bytes(b"unknown")

    preview = build_execution_preview(
        _plan(status=AssignmentStatus.NEEDS_REVIEW),
        {"file_001": _metadata(source)},
        tmp_path,
    )

    assert preview.operations == ()
    assert len(preview.skipped) == 1
    assert preview.skipped[0].source == source


@pytest.mark.parametrize(
    "unsafe_name",
    ["../逃逸", "子目录/逃逸", "子目录\\逃逸", ".."],
)
def test_preview_rejects_unsafe_category_names(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    """Model-generated category names must not control filesystem paths."""

    source = tmp_path / "report.txt"
    source.write_text("report")

    with pytest.raises(ExecutionPreviewError, match="unsafe name"):
        build_execution_preview(
            _plan(category_name=unsafe_name),
            {"file_001": _metadata(source)},
            tmp_path,
        )


def test_preview_rejects_a_source_outside_the_root(tmp_path: Path) -> None:
    """A plan must not move a file that was not directly inside the root."""

    root = tmp_path / "desktop"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")

    with pytest.raises(ExecutionPreviewError, match="outside the scan root"):
        build_execution_preview(
            _plan(),
            {"file_001": _metadata(outside)},
            root,
        )


def test_preview_refuses_to_overwrite_an_existing_file(
    tmp_path: Path,
) -> None:
    """An existing destination must stop preview generation."""

    source = tmp_path / "report.txt"
    source.write_text("new")
    destination_directory = tmp_path / "工作" / "项目资料"
    destination_directory.mkdir(parents=True)
    (destination_directory / source.name).write_text("existing")

    with pytest.raises(ExecutionPreviewError, match="would overwrite"):
        build_execution_preview(
            _plan(),
            {"file_001": _metadata(source)},
            tmp_path,
        )


def test_preview_rejects_a_symbolic_link_source(tmp_path: Path) -> None:
    """A symlink must not become a move source after scanning."""

    target = tmp_path / "target.txt"
    target.write_text("target")
    source = tmp_path / "link.txt"
    source.symlink_to(target)

    metadata = FileMetadata(
        name=source.name,
        path=source,
        extension=".txt",
        size_bytes=source.stat().st_size,
        created_at=None,
        modified_at=datetime.now(UTC),
    )

    with pytest.raises(ExecutionPreviewError, match="symbolic-link"):
        build_execution_preview(
            _plan(),
            {"file_001": metadata},
            tmp_path,
        )


def test_preview_rejects_a_file_changed_after_scanning(
    tmp_path: Path,
) -> None:
    """A file modified after metadata capture must not be moved."""

    source = tmp_path / "report.txt"
    source.write_text("old")
    metadata = _metadata(source)
    source.write_text("new content with a different size")

    with pytest.raises(ExecutionPreviewError, match="changed after scanning"):
        build_execution_preview(
            _plan(),
            {"file_001": metadata},
            tmp_path,
        )


def test_execute_preview_moves_a_file_and_creates_directories(
    tmp_path: Path,
) -> None:
    """A validated execution should create directories and move the file."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")
    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )

    result = execute_preview(preview, tmp_path)
    destination = tmp_path / "工作" / "项目资料" / source.name

    assert not source.exists()
    assert destination.read_bytes() == b"pdf"
    assert result.moved == preview.operations
    assert result.created_directories == (
        tmp_path / "工作",
        tmp_path / "工作" / "项目资料",
    )


def test_execute_preview_rechecks_for_new_destination_conflicts(
    tmp_path: Path,
) -> None:
    """A destination created after preview must prevent every move."""

    source = tmp_path / "report.txt"
    source.write_text("source")
    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )
    destination = preview.operations[0].destination
    destination.parent.mkdir(parents=True)
    destination.write_text("existing")

    with pytest.raises(ExecutionError, match="destination already exists"):
        execute_preview(preview, tmp_path)

    assert source.read_text() == "source"
    assert destination.read_text() == "existing"


def test_atomic_move_never_overwrites_an_existing_destination(
    tmp_path: Path,
) -> None:
    """The final move primitive must preserve both files on a conflict."""

    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("source")
    destination.write_text("existing")

    with pytest.raises(FileExistsError):
        executor_module._move_without_overwrite(source, destination)

    assert source.read_text() == "source"
    assert destination.read_text() == "existing"


def test_execute_preview_rejects_a_changed_source(tmp_path: Path) -> None:
    """A source changed after preview must not be moved."""

    source = tmp_path / "report.txt"
    source.write_text("old")
    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )
    source.write_text("new content")

    with pytest.raises(ExecutionError, match="changed after preview"):
        execute_preview(preview, tmp_path)

    assert source.read_text() == "new content"
    assert not (tmp_path / "工作").exists()


def test_execute_preview_rejects_a_forged_outside_destination(
    tmp_path: Path,
) -> None:
    """Even a manually forged preview cannot move outside the root."""

    source = tmp_path / "report.txt"
    source.write_text("report")
    source_stat = source.stat()
    outside = tmp_path.parent / "outside" / source.name
    preview = ExecutionPreview(
        operations=(
            MoveOperation(
                file_id="file_001",
                source=source,
                destination=outside,
                size_bytes=source_stat.st_size,
                modified_time_ns=source_stat.st_mtime_ns,
            ),
        ),
        skipped=(),
    )

    with pytest.raises(ExecutionError, match="outside the execution root"):
        execute_preview(preview, tmp_path)

    assert source.exists()
    assert not outside.exists()


def test_execute_preview_rolls_back_after_a_later_move_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a later move fails, earlier moves and directories are restored."""

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first")
    second.write_text("second")
    first_stat = first.stat()
    second_stat = second.stat()
    destination_directory = tmp_path / "工作"
    preview = ExecutionPreview(
        operations=(
            MoveOperation(
                file_id="file_001",
                source=first,
                destination=destination_directory / first.name,
                size_bytes=first_stat.st_size,
                modified_time_ns=first_stat.st_mtime_ns,
            ),
            MoveOperation(
                file_id="file_002",
                source=second,
                destination=destination_directory / second.name,
                size_bytes=second_stat.st_size,
                modified_time_ns=second_stat.st_mtime_ns,
            ),
        ),
        skipped=(),
    )
    original_move = executor_module._move_without_overwrite

    def fail_second_move(source: Path, destination: Path) -> None:
        if source == second:
            raise OSError("simulated move failure")
        original_move(source, destination)

    monkeypatch.setattr(
        executor_module,
        "_move_without_overwrite",
        fail_second_move,
    )

    with pytest.raises(ExecutionError, match="was rolled back"):
        execute_preview(preview, tmp_path)

    assert first.read_text() == "first"
    assert second.read_text() == "second"
    assert not destination_directory.exists()


def test_undo_last_execution_restores_file_and_removes_empty_directories(
    tmp_path: Path,
) -> None:
    """A completed run should be reversible after an app restart."""

    source = tmp_path / "report.txt"
    source.write_text("report")
    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )
    result = execute_preview(preview, tmp_path)
    destination = result.moved[0].destination

    assert get_last_execution_move_count(tmp_path) == 1
    undo_result = undo_last_execution(tmp_path)

    assert source.read_text() == "report"
    assert not destination.exists()
    assert not (tmp_path / "工作").exists()
    assert undo_result.restored == (source,)
    assert get_last_execution_move_count(tmp_path) is None


def test_undo_rejects_an_organized_file_modified_after_execution(
    tmp_path: Path,
) -> None:
    """Undo must not relocate content changed after the agent moved it."""

    source = tmp_path / "report.txt"
    source.write_text("original")
    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )
    result = execute_preview(preview, tmp_path)
    destination = result.moved[0].destination
    destination.write_text("changed after organization")

    with pytest.raises(ExecutionError, match="changed"):
        undo_last_execution(tmp_path)

    assert not source.exists()
    assert destination.read_text() == "changed after organization"
    assert (tmp_path / UNDO_JOURNAL_NAME).exists()


def test_undo_rejects_an_occupied_original_path(tmp_path: Path) -> None:
    """Undo must never overwrite a new file created at the old location."""

    source = tmp_path / "report.txt"
    source.write_text("organized")
    preview = build_execution_preview(
        _plan(),
        {"file_001": _metadata(source)},
        tmp_path,
    )
    result = execute_preview(preview, tmp_path)
    destination = result.moved[0].destination
    source.write_text("new user file")

    with pytest.raises(ExecutionError, match="occupied"):
        undo_last_execution(tmp_path)

    assert source.read_text() == "new user file"
    assert destination.read_text() == "organized"


def test_directory_container_moves_as_a_unit_and_can_be_undone(
    tmp_path: Path,
) -> None:
    """A classified top-level folder should remain intact across move and undo."""

    container = tmp_path / "20260001_示例学生_实验5"
    container.mkdir()
    (container / "实验报告.txt").write_text("计算机组成原理 CPU ALU")
    metadata = scan_directory(tmp_path)[0]
    assert metadata.is_directory is True
    preview = build_execution_preview(
        _plan(),
        {"file_001": metadata},
        tmp_path,
    )

    result = execute_preview(preview, tmp_path)
    destination = result.moved[0].destination

    assert not container.exists()
    assert (destination / "实验报告.txt").read_text() == (
        "计算机组成原理 CPU ALU"
    )
    undo_last_execution(tmp_path)
    assert (container / "实验报告.txt").read_text() == (
        "计算机组成原理 CPU ALU"
    )
    assert not destination.exists()


def test_executor_rejects_running_agent_project_even_if_scan_is_bypassed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution validation is a final defense against self-movement."""

    project = tmp_path / "desktop-agent"
    project.mkdir()
    (project / "app.py").write_text("application")
    metadata = scan_directory(tmp_path)[0]
    monkeypatch.setattr(scanner_module, "AGENT_PROJECT_ROOT", project)

    with pytest.raises(ExecutionPreviewError, match="protected"):
        build_execution_preview(
            _plan(),
            {"file_001": metadata},
            tmp_path,
        )
