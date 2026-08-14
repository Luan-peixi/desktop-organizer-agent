"""Cross-object validation for model-generated organization plans."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from desktop_agent.models import FileMetadata, OrganizationPlan


class PlanValidationError(ValueError):
    """Raised when an organization plan is structurally unsafe."""

    def __init__(self, issues: Iterable[str]) -> None:
        self.issues = tuple(issues)
        message = "Invalid organization plan:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


def index_files(
    files: Sequence[FileMetadata],
) -> dict[str, FileMetadata]:
    """Assign stable, prompt-safe identifiers to scanned files."""

    return {
        f"file_{position:03d}": file
        for position, file in enumerate(files, start=1)
    }


def validate_plan(
    plan: OrganizationPlan,
    files_by_id: Mapping[str, FileMetadata],
) -> OrganizationPlan:
    """Validate a model plan against the files that were actually scanned."""

    issues: list[str] = []
    expected_file_ids = set(files_by_id)
    assigned_file_ids = [
        assignment.file_id for assignment in plan.assignments
    ]
    assignment_counts = Counter(assigned_file_ids)

    duplicate_file_ids = sorted(
        file_id
        for file_id, count in assignment_counts.items()
        if count > 1
    )
    if duplicate_file_ids:
        issues.append(
            "duplicate file assignments: "
            + ", ".join(duplicate_file_ids)
        )

    missing_file_ids = sorted(expected_file_ids - set(assigned_file_ids))
    if missing_file_ids:
        issues.append(
            "missing file assignments: " + ", ".join(missing_file_ids)
        )

    unknown_file_ids = sorted(set(assigned_file_ids) - expected_file_ids)
    if unknown_file_ids:
        issues.append(
            "unknown file assignments: " + ", ".join(unknown_file_ids)
        )

    category_ids = [category.id for category in plan.categories]
    category_counts = Counter(category_ids)
    duplicate_category_ids = sorted(
        category_id
        for category_id, count in category_counts.items()
        if count > 1
    )
    if duplicate_category_ids:
        issues.append(
            "duplicate category IDs: " + ", ".join(duplicate_category_ids)
        )

    categories_by_id = {
        category.id: category for category in plan.categories
    }
    known_category_ids = set(categories_by_id)

    for category in plan.categories:
        if (
            category.parent_id is not None
            and category.parent_id not in known_category_ids
        ):
            issues.append(
                f"category {category.id!r} references unknown parent "
                f"{category.parent_id!r}"
            )

    for assignment in plan.assignments:
        if (
            assignment.category_id is not None
            and assignment.category_id not in known_category_ids
        ):
            issues.append(
                f"file {assignment.file_id!r} references unknown category "
                f"{assignment.category_id!r}"
            )

    _validate_category_hierarchy(categories_by_id, issues)

    if issues:
        raise PlanValidationError(issues)

    return plan


def _validate_category_hierarchy(
    categories_by_id: Mapping[str, object],
    issues: list[str],
) -> None:
    """Detect category cycles and hierarchies deeper than two levels."""

    for category_id in categories_by_id:
        current_id = category_id
        visited = {category_id}
        depth = 1

        while True:
            category = categories_by_id[current_id]
            parent_id = getattr(category, "parent_id")

            if parent_id is None or parent_id not in categories_by_id:
                break

            if parent_id in visited:
                issues.append(
                    f"category hierarchy contains a cycle at {parent_id!r}"
                )
                break

            visited.add(parent_id)
            depth += 1

            if depth > 2:
                issues.append(
                    f"category {category_id!r} exceeds the maximum depth of 2"
                )
                break

            current_id = parent_id
