"""Data models for files and organization plans."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContentExtractionStatus(StrEnum):
    """Outcome of the optional local content-extraction step."""

    NOT_REQUESTED = "not_requested"
    EXTRACTED = "extracted"
    UNSUPPORTED = "unsupported"
    TOO_LARGE = "too_large"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """A read-only snapshot of one file's metadata."""

    name: str
    path: Path
    extension: str
    size_bytes: int
    created_at: datetime | None
    modified_at: datetime
    is_directory: bool = False
    tree_fingerprint: str | None = None
    content_status: ContentExtractionStatus = (
        ContentExtractionStatus.NOT_REQUESTED
    )
    content_excerpt: str | None = None
    academic_material_hint: str | None = None
    course_hint: str | None = None
    course_subject: str | None = None
    course_confidence: float | None = None


class StrictModel(BaseModel):
    """Base model for immutable, strictly validated LLM output."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class AssignmentStatus(StrEnum):
    """Workflow status for a file assignment."""

    CLASSIFIED = "classified"
    NEEDS_REVIEW = "needs_review"


class CategoryDefinition(StrictModel):
    """A semantic category dynamically proposed by the model."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,49}$")
    name: str = Field(min_length=1, max_length=40)
    description: str = Field(min_length=1, max_length=200)
    parent_id: str | None


class FileAssignment(StrictModel):
    """The model's proposed category assignment for one scanned file."""

    file_id: str = Field(pattern=r"^file_[0-9]+$")
    category_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=300)
    status: AssignmentStatus

    @model_validator(mode="after")
    def classified_file_requires_category(self) -> "FileAssignment":
        """Require classified files to reference a proposed category."""

        if (
            self.status is AssignmentStatus.CLASSIFIED
            and self.category_id is None
        ):
            raise ValueError("A classified file must reference a category.")
        return self


class OrganizationPlan(StrictModel):
    """A complete dynamic organization plan returned by the model."""

    overview: str = Field(min_length=1, max_length=500)
    categories: list[CategoryDefinition] = Field(
        min_length=1,
        max_length=20,
    )
    assignments: list[FileAssignment] = Field(min_length=1)
