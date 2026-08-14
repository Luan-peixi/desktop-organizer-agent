"""Tests for GPT-backed natural-language operation planning."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from desktop_agent.instruction_agent import OpenAIInstructionPlanner
from desktop_agent.instruction_models import (
    InstructionAction,
    InstructionEntry,
    InstructionOperation,
    InstructionPlan,
)
from desktop_agent.model_client import ModelPlanError


class FakeResponses:
    def __init__(self, plan: InstructionPlan) -> None:
        self.plan = plan
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.plan)


class FakeClient:
    def __init__(self, plan: InstructionPlan) -> None:
        self.responses = FakeResponses(plan)


def _entry(entry_id: str, name: str, order: int) -> InstructionEntry:
    path = Path("/private/Desktop/实验5") / name
    return InstructionEntry(
        entry_id=entry_id,
        path=path,
        relative_path=f"实验5/{name}",
        name=name,
        parent_relative_path="实验5",
        display_order=order,
        depth=2,
        is_directory=False,
        extension=path.suffix.lower(),
        size_bytes=10,
        modified_time_ns=1,
    )


def test_specific_second_entry_plan_uses_structured_output() -> None:
    """The model may keep one sibling and move another by stable ID."""

    plan = InstructionPlan(
        summary="保留第一项，移动第二项。",
        clarification_required=False,
        operations=[
            InstructionOperation(
                entry_id="entry_0002",
                action=InstructionAction.KEEP,
                reason="用户要求保留第一项。",
            ),
            InstructionOperation(
                entry_id="entry_0003",
                action=InstructionAction.MOVE,
                destination_categories=["课程资料", "人工智能"],
                reason="用户指定第二项归入人工智能。",
            ),
        ],
    )
    client = FakeClient(plan)
    planner = OpenAIInstructionPlanner(client=client, model="gpt-test")
    entries = {
        "entry_0002": _entry("entry_0002", "a.txt", 1),
        "entry_0003": _entry("entry_0003", "b.txt", 2),
    }

    result = planner.generate_plan(
        [{"role": "user", "content": "第一个保留，第二个移到人工智能"}],
        entries,
    )

    assert result == plan
    request = client.responses.calls[0]
    assert request["text_format"] is InstructionPlan
    assert "/private/Desktop" not in str(request["input"])
    assert "display_order" in str(request["input"])


def test_hallucinated_entry_id_is_rejected() -> None:
    """GPT cannot operate on an entry absent from the inventory."""

    plan = InstructionPlan(
        summary="移动文件。",
        clarification_required=False,
        operations=[
            InstructionOperation(
                entry_id="entry_9999",
                action=InstructionAction.MOVE,
                destination_categories=["人工智能"],
                reason="用户要求。",
            )
        ],
    )
    planner = OpenAIInstructionPlanner(client=FakeClient(plan))

    with pytest.raises(ModelPlanError, match="不存在"):
        planner.generate_plan(
            [{"role": "user", "content": "移动它"}],
            {"entry_0001": _entry("entry_0001", "a.txt", 1)},
        )


def test_clarification_plan_contains_no_operations() -> None:
    """Ambiguity is represented as a question, never a guessed move."""

    plan = InstructionPlan(
        summary="需要明确是哪个文件夹。",
        clarification_required=True,
        clarification_question="你指的“第二个”是哪个文件夹中的第二项？",
        operations=[],
    )

    assert plan.clarification_required is True
    assert plan.operations == []


def test_moving_parent_while_keeping_child_is_rejected() -> None:
    """A keep action must not be silently defeated by moving its parent."""

    parent_path = Path("/private/Desktop/实验5")
    parent = InstructionEntry(
        entry_id="entry_0001",
        path=parent_path,
        relative_path="实验5",
        name="实验5",
        parent_relative_path=None,
        display_order=1,
        depth=1,
        is_directory=True,
        extension="",
        size_bytes=10,
        modified_time_ns=1,
    )
    child = _entry("entry_0002", "a.txt", 1)
    plan = InstructionPlan(
        summary="移动父文件夹但保留子文件。",
        clarification_required=False,
        operations=[
            InstructionOperation(
                entry_id="entry_0001",
                action=InstructionAction.MOVE,
                destination_categories=["人工智能"],
                reason="移动父文件夹。",
            ),
            InstructionOperation(
                entry_id="entry_0002",
                action=InstructionAction.KEEP,
                reason="保留子文件。",
            ),
        ],
    )

    with pytest.raises(ModelPlanError, match="父文件夹"):
        OpenAIInstructionPlanner(client=FakeClient(plan)).generate_plan(
            [{"role": "user", "content": "移动文件夹但保留里面的 a"}],
            {"entry_0001": parent, "entry_0002": child},
        )
