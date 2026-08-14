"""Tests for dynamic organization plan models."""

import pytest
from pydantic import ValidationError

from desktop_agent.models import (
    AssignmentStatus,
    CategoryDefinition,
    FileAssignment,
    OrganizationPlan,
)


def test_organization_plan_accepts_dynamic_categories() -> None:
    """A valid plan may contain categories created for the current files."""

    plan = OrganizationPlan(
        overview="文件主要与实习项目有关。",
        categories=[
            CategoryDefinition(
                id="internship_project",
                name="实习项目",
                description="实习期间产生的项目资料。",
                parent_id=None,
            )
        ],
        assignments=[
            FileAssignment(
                file_id="file_001",
                category_id="internship_project",
                confidence=0.94,
                reason="文件名表明它是项目复盘资料。",
                status=AssignmentStatus.CLASSIFIED,
            )
        ],
    )

    assert plan.categories[0].name == "实习项目"
    assert plan.assignments[0].category_id == "internship_project"


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_file_assignment_rejects_invalid_confidence(
    confidence: float,
) -> None:
    """Confidence must stay within the closed interval from zero to one."""

    with pytest.raises(ValidationError):
        FileAssignment(
            file_id="file_001",
            category_id="documents",
            confidence=confidence,
            reason="测试置信度范围。",
            status=AssignmentStatus.CLASSIFIED,
        )


def test_classified_assignment_requires_category() -> None:
    """A classified file cannot omit its category reference."""

    with pytest.raises(
        ValidationError,
        match="A classified file must reference a category",
    ):
        FileAssignment(
            file_id="file_001",
            category_id=None,
            confidence=0.9,
            reason="这个结果缺少类别。",
            status=AssignmentStatus.CLASSIFIED,
        )


def test_category_rejects_unsafe_id() -> None:
    """Category identifiers cannot contain path separators."""

    with pytest.raises(ValidationError):
        CategoryDefinition(
            id="../../outside",
            name="不安全类别",
            description="该 ID 不能被接受。",
            parent_id=None,
        )


def test_category_rejects_unexpected_fields() -> None:
    """Unexpected model output fields should not be silently ignored."""

    with pytest.raises(ValidationError):
        CategoryDefinition.model_validate(
            {
                "id": "documents",
                "name": "文档",
                "description": "普通文档。",
                "parent_id": None,
                "target_path": "/Users/example/Desktop",
            }
        )
