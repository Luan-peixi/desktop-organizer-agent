"""Tests for the command-line interface."""

from pathlib import Path

import pytest

from desktop_agent.cli import main
from desktop_agent.models import (
    AssignmentStatus,
    CategoryDefinition,
    FileAssignment,
    OrganizationPlan,
)


class FakePlanGenerator:
    """Return a prepared plan and record whether the model was requested."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_plan(self, files_by_id: object) -> OrganizationPlan:
        """Return a plan without calling an external model."""

        self.calls += 1
        return OrganizationPlan(
            overview="文件主要与实习项目有关。",
            categories=[
                CategoryDefinition(
                    id="work",
                    name="工作",
                    description="工作和实习相关文件。",
                    parent_id=None,
                ),
                CategoryDefinition(
                    id="internship",
                    name="实习项目",
                    description="当前实习项目产生的材料。",
                    parent_id="work",
                ),
            ],
            assignments=[
                FileAssignment(
                    file_id="file_001",
                    category_id="internship",
                    confidence=0.94,
                    reason="文件名表明它是项目复盘资料。",
                    status=AssignmentStatus.CLASSIFIED,
                )
            ],
        )


def test_main_displays_scanned_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A successful scan should display file metadata."""

    (tmp_path / "report.PDF").write_bytes(b"pdf")

    exit_code = main(["scan", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "发现 1 个文件" in captured.out
    assert "report.PDF" in captured.out
    assert "扩展名：.pdf" in captured.out
    assert "大小：3 B" in captured.out
    assert captured.err == ""


def test_main_reports_empty_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty directory should produce a successful, clear message."""

    exit_code = main(["scan", str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "没有发现可处理的文件。" in captured.out
    assert captured.err == ""


def test_main_reports_missing_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing directory should produce a Chinese error and exit code 2."""

    missing_directory = tmp_path / "missing"

    exit_code = main(["scan", str(missing_directory)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "错误：指定目录不存在" in captured.err
    assert str(missing_directory) in captured.err


def test_plan_command_displays_dynamic_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The plan command should display validated model suggestions."""

    (tmp_path / "项目复盘.PDF").write_bytes(b"pdf")
    generator = FakePlanGenerator()

    exit_code = main(
        ["plan", str(tmp_path)],
        plan_generator=generator,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert generator.calls == 1
    assert "整理计划：文件主要与实习项目有关。" in captured.out
    assert "动态类别：" in captured.out
    assert "实习项目" in captured.out
    assert "项目复盘.PDF" in captured.out
    assert "置信度：94%" in captured.out
    assert "状态：已分类" in captured.out
    assert captured.err == ""


def test_plan_command_skips_model_for_empty_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty directory should not consume an API request."""

    generator = FakePlanGenerator()

    exit_code = main(
        ["plan", str(tmp_path)],
        plan_generator=generator,
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert generator.calls == 0
    assert "没有发现可处理的文件，不调用模型。" in captured.out
    assert captured.err == ""


def test_preview_command_displays_paths_without_moving_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A preview should show its destination and leave the source untouched."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")
    generator = FakePlanGenerator()

    exit_code = main(
        ["preview", str(tmp_path)],
        plan_generator=generator,
    )
    captured = capsys.readouterr()

    expected_destination = (
        tmp_path.resolve() / "工作" / "实习项目" / source.name
    )
    assert exit_code == 0
    assert generator.calls == 1
    assert "执行预演（未创建目录、未移动文件）：" in captured.out
    assert f"来源：{source}" in captured.out
    assert f"目标：{expected_destination}" in captured.out
    assert source.exists()
    assert not (tmp_path / "工作").exists()
    assert captured.err == ""


def test_organize_command_cancels_without_exact_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Any confirmation other than exact MOVE must leave files untouched."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")

    exit_code = main(
        ["organize", str(tmp_path)],
        plan_generator=FakePlanGenerator(),
        confirmation_reader=lambda prompt: "move",
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "已取消，未更改任何文件。" in captured.out
    assert source.exists()
    assert not (tmp_path / "工作").exists()
    assert captured.err == ""


def test_organize_command_moves_only_after_exact_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exact MOVE should execute the same paths that were previewed."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")

    exit_code = main(
        ["organize", str(tmp_path)],
        plan_generator=FakePlanGenerator(),
        confirmation_reader=lambda prompt: "MOVE",
    )
    captured = capsys.readouterr()
    destination = tmp_path / "工作" / "实习项目" / source.name

    assert exit_code == 0
    assert "整理完成：成功移动 1 个文件。" in captured.out
    assert not source.exists()
    assert destination.read_bytes() == b"pdf"
    assert captured.err == ""


def test_organize_command_rejects_a_change_during_confirmation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file changed while the user reviews the preview must not move."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")

    def change_file_then_confirm(prompt: str) -> str:
        source.write_bytes(b"changed after preview")
        return "MOVE"

    exit_code = main(
        ["organize", str(tmp_path)],
        plan_generator=FakePlanGenerator(),
        confirmation_reader=change_file_then_confirm,
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert source.read_bytes() == b"changed after preview"
    assert not (tmp_path / "工作").exists()
    assert "changed after preview" in captured.err
