"""Strict models for natural-language, fine-grained organization requests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from desktop_agent.models import StrictModel


@dataclass(frozen=True, slots=True)
class InstructionEntry:
    """A safe snapshot of one selected top-level or nested entry."""

    entry_id: str
    path: Path
    relative_path: str
    name: str
    parent_relative_path: str | None
    display_order: int
    depth: int
    is_directory: bool
    extension: str
    size_bytes: int
    modified_time_ns: int
    tree_fingerprint: str | None = None


class InstructionAction(StrEnum):
    """The only file actions the preference Agent may propose."""

    KEEP = "keep"
    MOVE = "move"


class InstructionOperation(StrictModel):
    """One model-proposed action referencing a real entry ID."""

    entry_id: str = Field(pattern=r"^entry_[0-9]+$")
    action: InstructionAction
    destination_categories: list[str] = Field(default_factory=list, max_length=2)
    reason: str = Field(min_length=1, max_length=300)

    @model_validator(mode="after")
    def destination_matches_action(self) -> "InstructionOperation":
        """Moves require a category, while keep actions cannot have one."""

        if self.action is InstructionAction.MOVE and not self.destination_categories:
            raise ValueError("A move operation requires a destination category.")
        if self.action is InstructionAction.KEEP and self.destination_categories:
            raise ValueError("A keep operation cannot have a destination category.")
        return self


class InstructionPlan(StrictModel):
    """A structured interpretation of one natural-language request."""

    summary: str = Field(min_length=1, max_length=500)
    clarification_required: bool
    clarification_question: str | None = Field(default=None, max_length=500)
    operations: list[InstructionOperation] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def clarification_is_consistent(self) -> "InstructionPlan":
        """An unresolved plan must ask rather than proposing filesystem changes."""

        if self.clarification_required:
            if not self.clarification_question:
                raise ValueError("Clarification requires a question.")
            if self.operations:
                raise ValueError("An unresolved plan cannot include operations.")
        elif self.clarification_question is not None:
            raise ValueError("A resolved plan cannot include a clarification question.")
        return self
