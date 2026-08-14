"""Tests for organization plan validation."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from desktop_agent.models import (
    AssignmentStatus,
    CategoryDefinition,
    FileAssignment,
    FileMetadata,
    OrganizationPlan,
)
from desktop_agent.plan_validator import (
    PlanValidationError,
    index_files,
    validate_plan,
)


def make_file(name: str) -> FileMetadata:
    """Create file metadata without touching the filesystem."""

    timestamp = datetime(2026, 7, 30, tzinfo=UTC)
    return FileMetadata(
        name=name,
        path=Path("/safe/demo") / name,
        extension=Path(name).suffix.lower(),
        size_bytes=100,
        created_at=timestamp,
        modified_at=timestamp,
    )


def make_assignment(
    file_id: str,
    category_id: str = "work",
) -> FileAssignment:
    """Create a classified assignment for validator tests."""

    return FileAssignment(
        file_id=file_id,
        category_id=category_id,
        confidence=0.9,
        reason="测试分类理由。",
        status=AssignmentStatus.CLASSIFIED,
    )


def make_plan(
    categories: list[CategoryDefinition],
    assignments: list[FileAssignment],
) -> OrganizationPlan:
    """Create a plan with supplied categories and assignments."""

    return OrganizationPlan(
        overview="测试整理计划。",
        categories=categories,
        assignments=assignments,
    )


def test_index_files_assigns_stable_safe_ids() -> None:
    """Scan order should map deterministically to prompt-safe IDs."""

    files = [make_file("a.txt"), make_file("b.pdf")]

    files_by_id = index_files(files)

    assert list(files_by_id) == ["file_001", "file_002"]
    assert files_by_id["file_001"].name == "a.txt"
    assert files_by_id["file_002"].name == "b.pdf"


def test_validate_plan_accepts_complete_two_level_plan() -> None:
    """A complete plan with a two-level hierarchy should pass."""

    files_by_id = index_files(
        [make_file("复盘.pdf"), make_file("代码.zip")]
    )
    categories = [
        CategoryDefinition(
            id="work",
            name="工作",
            description="工作相关文件。",
            parent_id=None,
        ),
        CategoryDefinition(
            id="internship",
            name="实习项目",
            description="实习项目相关文件。",
            parent_id="work",
        ),
    ]
    plan = make_plan(
        categories,
        [
            make_assignment("file_001", "internship"),
            make_assignment("file_002", "internship"),
        ],
    )

    assert validate_plan(plan, files_by_id) is plan


def test_validate_plan_rejects_missing_duplicate_and_unknown_files() -> None:
    """Every scanned file must be assigned exactly once."""

    files_by_id = index_files(
        [make_file("a.txt"), make_file("b.txt")]
    )
    plan = make_plan(
        [
            CategoryDefinition(
                id="work",
                name="工作",
                description="工作文件。",
                parent_id=None,
            )
        ],
        [
            make_assignment("file_001"),
            make_assignment("file_001"),
            make_assignment("file_999"),
        ],
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(plan, files_by_id)

    message = str(error.value)
    assert "duplicate file assignments: file_001" in message
    assert "missing file assignments: file_002" in message
    assert "unknown file assignments: file_999" in message


def test_validate_plan_rejects_invalid_category_references() -> None:
    """Parent and assignment references must point to real categories."""

    files_by_id = index_files([make_file("a.txt")])
    plan = make_plan(
        [
            CategoryDefinition(
                id="child",
                name="子类别",
                description="父类别不存在。",
                parent_id="missing_parent",
            )
        ],
        [make_assignment("file_001", "missing_category")],
    )

    with pytest.raises(PlanValidationError) as error:
        validate_plan(plan, files_by_id)

    message = str(error.value)
    assert "references unknown parent 'missing_parent'" in message
    assert "references unknown category 'missing_category'" in message


def test_validate_plan_rejects_duplicate_category_ids() -> None:
    """Category identifiers must be unique."""

    files_by_id = index_files([make_file("a.txt")])
    duplicate_categories = [
        CategoryDefinition(
            id="work",
            name="工作一",
            description="第一个定义。",
            parent_id=None,
        ),
        CategoryDefinition(
            id="work",
            name="工作二",
            description="第二个定义。",
            parent_id=None,
        ),
    ]
    plan = make_plan(
        duplicate_categories,
        [make_assignment("file_001")],
    )

    with pytest.raises(
        PlanValidationError,
        match="duplicate category IDs: work",
    ):
        validate_plan(plan, files_by_id)


@pytest.mark.parametrize(
    ("categories", "expected_message"),
    [
        (
            [
                CategoryDefinition(
                    id="first",
                    name="第一层",
                    description="循环起点。",
                    parent_id="second",
                ),
                CategoryDefinition(
                    id="second",
                    name="第二层",
                    description="循环终点。",
                    parent_id="first",
                ),
            ],
            "category hierarchy contains a cycle",
        ),
        (
            [
                CategoryDefinition(
                    id="root",
                    name="根类别",
                    description="第一层。",
                    parent_id=None,
                ),
                CategoryDefinition(
                    id="child",
                    name="子类别",
                    description="第二层。",
                    parent_id="root",
                ),
                CategoryDefinition(
                    id="grandchild",
                    name="孙类别",
                    description="不允许的第三层。",
                    parent_id="child",
                ),
            ],
            "exceeds the maximum depth of 2",
        ),
    ],
)
def test_validate_plan_rejects_unsafe_hierarchy(
    categories: list[CategoryDefinition],
    expected_message: str,
) -> None:
    """Category graphs cannot contain cycles or exceed two levels."""

    files_by_id = index_files([make_file("a.txt")])
    plan = make_plan(
        categories,
        [make_assignment("file_001", categories[0].id)],
    )

    with pytest.raises(PlanValidationError, match=expected_message):
        validate_plan(plan, files_by_id)
