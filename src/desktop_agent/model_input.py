"""Build safe, structured metadata payloads for the language model."""

import json
from collections.abc import Mapping
from typing import TypedDict

from desktop_agent.models import FileMetadata


class FilePromptRecord(TypedDict):
    """A path-free representation of one scanned file."""

    file_id: str
    name: str
    extension: str
    size_bytes: int
    created_at: str | None
    modified_at: str
    entry_type: str
    auto_move_eligible: bool
    content_status: str
    content_excerpt: str | None
    academic_material_hint: str | None
    course_hint: str | None
    course_subject: str | None
    course_confidence: float | None


def build_file_records(
    files_by_id: Mapping[str, FileMetadata],
) -> list[FilePromptRecord]:
    """Convert file snapshots into records safe to send to a model."""

    return [
        FilePromptRecord(
            file_id=file_id,
            name=file.name,
            extension=file.extension,
            size_bytes=file.size_bytes,
            created_at=(
                file.created_at.isoformat()
                if file.created_at is not None
                else None
            ),
            modified_at=file.modified_at.isoformat(),
            entry_type="directory" if file.is_directory else "file",
            auto_move_eligible=(
                not file.is_directory or file.tree_fingerprint is not None
            ),
            content_status=file.content_status.value,
            content_excerpt=file.content_excerpt,
            academic_material_hint=file.academic_material_hint,
            course_hint=file.course_hint,
            course_subject=file.course_subject,
            course_confidence=file.course_confidence,
        )
        for file_id, file in files_by_id.items()
    ]


def serialize_files_for_model(
    files_by_id: Mapping[str, FileMetadata],
) -> str:
    """Serialize safe file records as readable, Unicode-preserving JSON."""

    payload = {"files": build_file_records(files_by_id)}
    return json.dumps(payload, ensure_ascii=False, indent=2)
