"""Tests for nested inventory, preview, execution, and undo."""

from pathlib import Path

from desktop_agent.executor import execute_preview, undo_last_execution
from desktop_agent.instruction_executor import build_instruction_preview
from desktop_agent.instruction_models import (
    InstructionAction,
    InstructionOperation,
    InstructionPlan,
)
from desktop_agent.instruction_scanner import (
    scan_instruction_entries,
    scan_instruction_paths,
)


def test_second_nested_entry_moves_and_undoes_to_original_folder(
    tmp_path: Path,
) -> None:
    """A specific child can move while its sibling stays, then be restored."""

    container = tmp_path / "20260001_示例学生_实验5"
    container.mkdir()
    first = container / "01_朴素贝叶斯.txt"
    second = container / "02_多层感知机.txt"
    first.write_text("first")
    second.write_text("second")
    entries = scan_instruction_entries(tmp_path, [container.name])
    children = [
        entry
        for entry in entries.values()
        if entry.parent_relative_path == container.name
    ]
    assert [entry.name for entry in children] == [first.name, second.name]
    assert [entry.display_order for entry in children] == [1, 2]

    plan = InstructionPlan(
        summary="保留第一个，移动第二个。",
        clarification_required=False,
        operations=[
            InstructionOperation(
                entry_id=children[0].entry_id,
                action=InstructionAction.KEEP,
                reason="保留第一个。",
            ),
            InstructionOperation(
                entry_id=children[1].entry_id,
                action=InstructionAction.MOVE,
                destination_categories=["课程资料", "人工智能"],
                reason="归入人工智能。",
            ),
        ],
    )
    preview = build_instruction_preview(plan, entries, tmp_path)

    assert len(preview.operations) == 1
    assert preview.operations[0].source == second
    assert preview.operations[0].destination == (
        tmp_path / "课程资料" / "人工智能" / second.name
    )
    execute_preview(preview, tmp_path)

    assert first.read_text() == "first"
    assert not second.exists()
    assert preview.operations[0].destination.read_text() == "second"

    result = undo_last_execution(tmp_path)

    assert result.restored == (second,)
    assert second.read_text() == "second"
    assert first.read_text() == "first"


def test_dependency_directories_are_not_exposed_to_instruction_agent(
    tmp_path: Path,
) -> None:
    """Virtual environments should not flood the conversational inventory."""

    container = tmp_path / "project"
    (container / ".venv" / "lib").mkdir(parents=True)
    (container / ".venv" / "lib" / "secret.py").write_text("ignored")
    (container / "report.pdf").write_text("report")

    entries = scan_instruction_entries(tmp_path, [container.name])

    assert any(entry.name == "report.pdf" for entry in entries.values())
    assert all(".venv" not in entry.relative_path for entry in entries.values())


def test_organized_result_can_be_scanned_and_moved_again(
    tmp_path: Path,
) -> None:
    """A user may refine an organized result into a different category."""

    current = tmp_path / "课程资料" / "人工智能" / "report.pdf"
    current.parent.mkdir(parents=True)
    current.write_text("machine learning report")

    entries = scan_instruction_paths(tmp_path, [current])
    entry = next(iter(entries.values()))
    plan = InstructionPlan(
        summary="改放到机器学习。",
        clarification_required=False,
        operations=[
            InstructionOperation(
                entry_id=entry.entry_id,
                action=InstructionAction.MOVE,
                destination_categories=["课程资料", "机器学习"],
                reason="用户对整理结果进行二次调整。",
            )
        ],
    )

    preview = build_instruction_preview(plan, entries, tmp_path)
    result = execute_preview(preview, tmp_path)

    adjusted = tmp_path / "课程资料" / "机器学习" / "report.pdf"
    assert result.moved[0].destination == adjusted
    assert adjusted.read_text() == "machine learning report"
    assert not current.exists()

    undo_last_execution(tmp_path)

    assert current.read_text() == "machine learning report"
    assert not adjusted.exists()
