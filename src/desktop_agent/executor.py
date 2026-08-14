"""Build safe, read-only previews for model-generated move operations."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
import unicodedata

from desktop_agent.entry_snapshot import SnapshotLimitError, snapshot_directory
from desktop_agent.models import (
    AssignmentStatus,
    CategoryDefinition,
    FileMetadata,
    OrganizationPlan,
)
from desktop_agent.plan_validator import validate_plan
from desktop_agent.scanner import is_protected_agent_path


class ExecutionPreviewError(ValueError):
    """Raised when a plan cannot be converted into safe move operations."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = tuple(issues)
        message = "Unsafe execution preview:\n- " + "\n- ".join(self.issues)
        super().__init__(message)


class ExecutionError(RuntimeError):
    """Raised when safe execution fails or cannot be rolled back fully."""


@dataclass(frozen=True, slots=True)
class MoveOperation:
    """One proposed file move that has not been executed."""

    file_id: str
    source: Path
    destination: Path
    size_bytes: int
    modified_time_ns: int
    is_directory: bool = False
    tree_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class SkippedFile:
    """A file deliberately excluded from automatic execution."""

    file_id: str
    source: Path
    reason: str


@dataclass(frozen=True, slots=True)
class ExecutionPreview:
    """A read-only set of proposed moves and deliberately skipped files."""

    operations: tuple[MoveOperation, ...]
    skipped: tuple[SkippedFile, ...]


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The filesystem changes completed by a successful execution."""

    moved: tuple[MoveOperation, ...]
    created_directories: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class UndoResult:
    """The files restored by a completed post-execution undo."""

    restored: tuple[Path, ...]
    removed_directories: tuple[Path, ...]


UNDO_JOURNAL_NAME = ".desktop_agent_last_execution.json"
CATEGORY_MARKER_NAME = ".desktop_agent_category"


def build_execution_preview(
    plan: OrganizationPlan,
    files_by_id: Mapping[str, FileMetadata],
    root: Path,
) -> ExecutionPreview:
    """Translate a validated model plan into safe, non-mutating operations."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved_root}")

    validate_plan(plan, files_by_id)

    categories_by_id = {
        category.id: category for category in plan.categories
    }
    category_paths: dict[str, tuple[str, ...]] = {}
    issues: list[str] = []

    for category in plan.categories:
        components = _category_components(category, categories_by_id)
        for component in components:
            issue = _validate_path_component(component)
            if issue is not None:
                issues.append(
                    f"category {category.id!r} has unsafe name "
                    f"{component!r}: {issue}"
                )
        category_paths[category.id] = components

    operations: list[MoveOperation] = []
    skipped: list[SkippedFile] = []
    destination_keys: set[str] = set()

    for assignment in plan.assignments:
        file = files_by_id[assignment.file_id]

        if assignment.status is AssignmentStatus.NEEDS_REVIEW:
            skipped.append(
                SkippedFile(
                    file_id=assignment.file_id,
                    source=file.path,
                    reason="模型标记为需要人工确认",
                )
            )
            continue

        if file.is_directory and file.tree_fingerprint is None:
            skipped.append(
                SkippedFile(
                    file_id=assignment.file_id,
                    source=file.path,
                    reason="文件夹过大，无法建立完整安全快照",
                )
            )
            continue

        source_issue = _validate_source(file, resolved_root)
        if source_issue is not None:
            issues.append(f"file {assignment.file_id!r}: {source_issue}")
            continue

        if assignment.category_id is None:
            issues.append(
                f"file {assignment.file_id!r} has no destination category"
            )
            continue

        components = category_paths[assignment.category_id]
        destination_parent = resolved_root.joinpath(*components)
        destination = destination_parent / file.name

        parent_issue = _validate_destination_parent(
            resolved_root,
            components,
        )
        if parent_issue is not None:
            issues.append(
                f"file {assignment.file_id!r}: {parent_issue}"
            )
            continue

        if destination.exists() or destination.is_symlink():
            issues.append(
                f"file {assignment.file_id!r} would overwrite existing "
                f"path {destination}"
            )
            continue

        destination_key = unicodedata.normalize(
            "NFKC",
            str(destination),
        ).casefold()
        if destination_key in destination_keys:
            issues.append(
                f"file {assignment.file_id!r} duplicates destination "
                f"{destination}"
            )
            continue

        destination_keys.add(destination_key)
        source_stat = file.path.stat(follow_symlinks=False)
        operations.append(
            MoveOperation(
                file_id=assignment.file_id,
                source=file.path,
                destination=destination,
                size_bytes=(
                    file.size_bytes if file.is_directory else source_stat.st_size
                ),
                modified_time_ns=source_stat.st_mtime_ns,
                is_directory=file.is_directory,
                tree_fingerprint=file.tree_fingerprint,
            )
        )

    if issues:
        raise ExecutionPreviewError(issues)

    return ExecutionPreview(
        operations=tuple(operations),
        skipped=tuple(skipped),
    )


def execute_preview(
    preview: ExecutionPreview,
    root: Path,
) -> ExecutionResult:
    """Execute a preview atomically where possible, rolling back on failure."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved_root}")

    issues = _validate_operations_for_execution(preview, resolved_root)
    if issues:
        raise ExecutionError(
            "Execution rejected:\n- " + "\n- ".join(issues)
        )

    moved: list[MoveOperation] = []
    created_directories: list[Path] = []

    try:
        for operation in preview.operations:
            relative_destination = operation.destination.relative_to(
                resolved_root
            )
            _create_destination_directories(
                resolved_root,
                relative_destination.parts[:-1],
                created_directories,
            )

            issue = _validate_operation_for_execution(
                operation,
                resolved_root,
            )
            if issue is not None:
                raise OSError(issue)

            _move_without_overwrite(
                operation.source,
                operation.destination,
            )
            moved.append(operation)
        _write_undo_journal(
            resolved_root,
            moved,
            created_directories,
        )
    except OSError as error:
        rollback_issues = _rollback(moved, created_directories)
        message = f"Execution failed and was rolled back: {error}"
        if rollback_issues:
            message += "\nRollback also failed:\n- " + "\n- ".join(
                rollback_issues
            )
        raise ExecutionError(message) from error

    return ExecutionResult(
        moved=tuple(moved),
        created_directories=tuple(created_directories),
    )


def get_last_execution_move_count(root: Path) -> int | None:
    """Return the undoable move count, or None when no journal exists."""

    resolved_root = root.resolve(strict=True)
    journal_path = resolved_root / UNDO_JOURNAL_NAME
    if not journal_path.exists():
        return None
    record = _read_undo_journal(resolved_root)
    return len(record["operations"])


def undo_last_execution(root: Path) -> UndoResult:
    """Safely reverse the most recently recorded successful execution."""

    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved_root}")

    record = _read_undo_journal(resolved_root)
    operations = _undo_operations(record, resolved_root)
    created_directories = _undo_directories(record, resolved_root)
    issues = _validate_undo_operations(operations, resolved_root)
    if issues:
        raise ExecutionError(
            "Undo rejected:\n- " + "\n- ".join(issues)
        )

    restored_operations: list[MoveOperation] = []
    try:
        for operation in reversed(operations):
            _move_without_overwrite(
                operation.destination,
                operation.source,
            )
            restored_operations.append(operation)
    except OSError as error:
        rollback_issues: list[str] = []
        for operation in reversed(restored_operations):
            try:
                _move_without_overwrite(
                    operation.source,
                    operation.destination,
                )
            except OSError as rollback_error:
                rollback_issues.append(
                    f"could not restore organized state for "
                    f"{operation.source.name}: {rollback_error}"
                )
        message = f"Undo failed and was rolled back: {error}"
        if rollback_issues:
            message += "\nUndo rollback also failed:\n- " + "\n- ".join(
                rollback_issues
            )
        raise ExecutionError(message) from error

    removed_directories: list[Path] = []
    for directory in reversed(created_directories):
        marker = directory / CATEGORY_MARKER_NAME
        try:
            if marker.is_file() and not marker.is_symlink():
                marker.unlink()
        except OSError:
            pass
        try:
            directory.rmdir()
            removed_directories.append(directory)
        except FileNotFoundError:
            continue
        except OSError:
            # A directory may now contain unrelated user files. Never remove it
            # recursively and never treat that as a failed file restoration.
            continue

    try:
        (resolved_root / UNDO_JOURNAL_NAME).unlink()
    except OSError as error:
        raise ExecutionError(
            "Files were restored, but the undo journal could not be removed: "
            f"{error}"
        ) from error

    return UndoResult(
        restored=tuple(operation.source for operation in operations),
        removed_directories=tuple(removed_directories),
    )


def _category_components(
    category: CategoryDefinition,
    categories_by_id: Mapping[str, CategoryDefinition],
) -> tuple[str, ...]:
    """Return the one- or two-level display path for a category."""

    if category.parent_id is None:
        return (category.name,)

    parent = categories_by_id[category.parent_id]
    return (parent.name, category.name)


def _validate_path_component(component: str) -> str | None:
    """Reject a model-proposed folder name that could alter the path."""

    normalized = unicodedata.normalize("NFKC", component)
    if normalized in {"", ".", ".."}:
        return "empty and dot-only folder names are forbidden"
    if "/" in normalized or "\\" in normalized:
        return "path separators are forbidden"
    if any(ord(character) < 32 for character in normalized):
        return "control characters are forbidden"
    return None


def _validate_source(file: FileMetadata, root: Path) -> str | None:
    """Ensure the source is still a direct, regular child of the scan root."""

    source = file.path
    if is_protected_agent_path(source):
        return "the running Agent project is protected from self-movement"
    if source.name != file.name:
        return "metadata name no longer matches the source path"
    if source.is_symlink():
        return "symbolic-link sources are forbidden"
    if not source.exists():
        return "source file no longer exists"
    if file.is_directory:
        if not source.is_dir():
            return "source is no longer a directory"
        if file.tree_fingerprint is None:
            return "directory is too large for a safe automatic snapshot"
        try:
            size_bytes, fingerprint = snapshot_directory(source)
        except (OSError, SnapshotLimitError):
            return "directory cannot be safely snapshotted"
        if size_bytes != file.size_bytes or fingerprint != file.tree_fingerprint:
            return "directory contents changed after scanning"
    elif not source.is_file():
        return "source is no longer a regular file"

    try:
        source_parent = source.parent.resolve(strict=True)
    except OSError:
        return "source parent cannot be resolved"

    if source_parent != root:
        return "source is outside the scan root"

    if not file.is_directory:
        source_stat = source.stat(follow_symlinks=False)
        if source_stat.st_size != file.size_bytes:
            return "source size changed after scanning"
        if abs(source_stat.st_mtime - file.modified_at.timestamp()) > 0.001:
            return "source modification time changed after scanning"
    return None


def _validate_destination_parent(
    root: Path,
    components: tuple[str, ...],
) -> str | None:
    """Reject existing symlinks or files in the destination hierarchy."""

    current = root
    for component in components:
        current = current / component
        if current.is_symlink():
            return f"destination folder is a symbolic link: {current}"
        if current.exists() and not current.is_dir():
            return f"destination folder path is occupied by a file: {current}"
    return None


def _validate_operations_for_execution(
    preview: ExecutionPreview,
    root: Path,
) -> list[str]:
    """Validate every operation before making any filesystem changes."""

    issues: list[str] = []
    source_keys: set[str] = set()
    destination_keys: set[str] = set()

    for operation in preview.operations:
        issue = _validate_operation_for_execution(operation, root)
        if issue is not None:
            issues.append(f"file {operation.file_id!r}: {issue}")

        source_key = _path_key(operation.source)
        if source_key in source_keys:
            issues.append(
                f"file {operation.file_id!r} duplicates source "
                f"{operation.source}"
            )
        source_keys.add(source_key)

        destination_key = _path_key(operation.destination)
        if destination_key in destination_keys:
            issues.append(
                f"file {operation.file_id!r} duplicates destination "
                f"{operation.destination}"
            )
        destination_keys.add(destination_key)

    return issues


def _validate_operation_for_execution(
    operation: MoveOperation,
    root: Path,
) -> str | None:
    """Recheck one operation immediately before it is executed."""

    source = operation.source
    if is_protected_agent_path(source):
        return "the running Agent project is protected from self-movement"
    if source.is_symlink():
        return "symbolic-link sources are forbidden"
    if not source.exists():
        return "source file no longer exists"
    if operation.is_directory:
        if not source.is_dir():
            return "source is no longer a directory"
        if operation.tree_fingerprint is None:
            return "directory has no safe tree fingerprint"
        try:
            size_bytes, fingerprint = snapshot_directory(source)
        except (OSError, SnapshotLimitError):
            return "directory cannot be safely snapshotted"
        if size_bytes != operation.size_bytes:
            return "directory size changed after preview"
        if fingerprint != operation.tree_fingerprint:
            return "directory contents changed after preview"
    elif not source.is_file():
        return "source is no longer a regular file"

    try:
        relative_source = source.relative_to(root)
    except ValueError:
        return "source is outside the execution root"
    if not relative_source.parts or len(relative_source.parts) > 4:
        return "source exceeds the allowed depth"
    current_source_parent = root
    for component in relative_source.parts[:-1]:
        current_source_parent = current_source_parent / component
        if current_source_parent.is_symlink():
            return f"source parent is a symbolic link: {current_source_parent}"

    if not operation.is_directory:
        source_stat = source.stat(follow_symlinks=False)
        if source_stat.st_size != operation.size_bytes:
            return "source size changed after preview"
        if source_stat.st_mtime_ns != operation.modified_time_ns:
            return "source modification time changed after preview"

    destination = operation.destination
    if not destination.is_absolute():
        return "destination must be an absolute path"

    try:
        relative_destination = destination.relative_to(root)
    except ValueError:
        return "destination is outside the execution root"

    if len(relative_destination.parts) < 2:
        return "destination must be inside a category folder"
    if any(part in {"", ".", ".."} for part in relative_destination.parts):
        return "destination contains unsafe path components"
    if destination.name != source.name:
        return "destination filename differs from source filename"
    try:
        destination.relative_to(source)
    except ValueError:
        pass
    else:
        return "destination cannot be inside the source"

    parent_issue = _validate_destination_parent(
        root,
        relative_destination.parts[:-1],
    )
    if parent_issue is not None:
        return parent_issue
    if destination.exists() or destination.is_symlink():
        return f"destination already exists: {destination}"
    return None


def _create_destination_directories(
    root: Path,
    components: tuple[str, ...],
    created_directories: list[Path],
) -> None:
    """Create each checked destination directory without following symlinks."""

    current = root
    for component in components:
        current = current / component
        if current.is_symlink():
            raise OSError(f"destination folder became a symlink: {current}")
        if current.exists():
            if not current.is_dir():
                raise OSError(
                    f"destination folder path is occupied: {current}"
                )
            continue

        current.mkdir()
        created_directories.append(current)
        (current / CATEGORY_MARKER_NAME).write_text(
            "Created by Desktop Organizer Agent.\n",
            encoding="utf-8",
        )


def _write_undo_journal(
    root: Path,
    moved: list[MoveOperation],
    created_directories: list[Path],
) -> None:
    """Atomically persist relative paths for the latest successful run."""

    payload = {
        "version": 1,
        "operations": [
            {
                "file_id": operation.file_id,
                "source": str(operation.source.relative_to(root)),
                "destination": str(operation.destination.relative_to(root)),
                "size_bytes": operation.size_bytes,
                "modified_time_ns": operation.modified_time_ns,
                "is_directory": operation.is_directory,
                "tree_fingerprint": operation.tree_fingerprint,
            }
            for operation in moved
        ],
        "created_directories": [
            str(directory.relative_to(root))
            for directory in created_directories
        ],
    }
    journal_path = root / UNDO_JOURNAL_NAME
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".desktop_agent_journal_",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, journal_path)
    except OSError:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass
        raise


def _read_undo_journal(root: Path) -> dict[str, object]:
    """Load and minimally validate the local undo journal schema."""

    journal_path = root / UNDO_JOURNAL_NAME
    if journal_path.is_symlink():
        raise ExecutionError("Undo journal must not be a symbolic link.")
    if not journal_path.exists():
        raise ExecutionError("No undoable execution record was found.")
    if not journal_path.is_file():
        raise ExecutionError("Undo journal is not a regular file.")
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutionError("Undo journal is unreadable or invalid.") from error
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ExecutionError("Undo journal has an unsupported format.")
    if not isinstance(payload.get("operations"), list):
        raise ExecutionError("Undo journal has no valid operations list.")
    if not isinstance(payload.get("created_directories"), list):
        raise ExecutionError("Undo journal has no valid directory list.")
    return payload


def _safe_relative_parts(value: object) -> tuple[str, ...] | None:
    """Return safe portable relative components from journal data."""

    if not isinstance(value, str):
        return None
    path = Path(value)
    if path.is_absolute() or not path.parts:
        return None
    if any(_validate_path_component(part) is not None for part in path.parts):
        return None
    return path.parts


def _undo_operations(
    record: dict[str, object],
    root: Path,
) -> tuple[MoveOperation, ...]:
    """Convert untrusted journal records into bounded move operations."""

    raw_operations = record["operations"]
    assert isinstance(raw_operations, list)
    operations: list[MoveOperation] = []
    for index, raw in enumerate(raw_operations, start=1):
        if not isinstance(raw, dict):
            raise ExecutionError("Undo journal contains an invalid operation.")
        source_parts = _safe_relative_parts(raw.get("source"))
        destination_parts = _safe_relative_parts(raw.get("destination"))
        size_bytes = raw.get("size_bytes")
        modified_time_ns = raw.get("modified_time_ns")
        is_directory = raw.get("is_directory", False)
        tree_fingerprint = raw.get("tree_fingerprint")
        file_id = raw.get("file_id")
        if (
            source_parts is None
            or len(source_parts) not in {1, 2, 3, 4}
            or destination_parts is None
            or len(destination_parts) not in {2, 3}
            or source_parts[-1] != destination_parts[-1]
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(modified_time_ns, int)
            or modified_time_ns < 0
            or not isinstance(file_id, str)
            or not isinstance(is_directory, bool)
            or (
                is_directory
                and (
                    not isinstance(tree_fingerprint, str)
                    or not tree_fingerprint
                )
            )
        ):
            raise ExecutionError(
                f"Undo journal operation {index} failed schema validation."
            )
        operations.append(
            MoveOperation(
                file_id=file_id,
                source=root.joinpath(*source_parts),
                destination=root.joinpath(*destination_parts),
                size_bytes=size_bytes,
                modified_time_ns=modified_time_ns,
                is_directory=is_directory,
                tree_fingerprint=(
                    tree_fingerprint
                    if isinstance(tree_fingerprint, str)
                    else None
                ),
            )
        )
    if not operations:
        raise ExecutionError("Undo journal contains no file moves.")
    return tuple(operations)


def _undo_directories(
    record: dict[str, object],
    root: Path,
) -> tuple[Path, ...]:
    """Convert safe one- or two-level created directory records."""

    raw_directories = record["created_directories"]
    assert isinstance(raw_directories, list)
    directories: list[Path] = []
    for raw in raw_directories:
        parts = _safe_relative_parts(raw)
        if parts is None or len(parts) not in {1, 2}:
            raise ExecutionError(
                "Undo journal contains an unsafe directory path."
            )
        directories.append(root.joinpath(*parts))
    return tuple(directories)


def _validate_undo_operations(
    operations: tuple[MoveOperation, ...],
    root: Path,
) -> list[str]:
    """Reject the whole undo before changing any file."""

    issues: list[str] = []
    source_keys: set[str] = set()
    destination_keys: set[str] = set()
    for operation in operations:
        source = operation.source
        destination = operation.destination
        if source.exists() or source.is_symlink():
            issues.append(f"original path is occupied: {source}")
        if destination.is_symlink():
            issues.append(f"organized file became a symbolic link: {destination}")
            continue
        if not destination.exists():
            issues.append(f"organized file is missing: {destination}")
            continue
        if operation.is_directory:
            if not destination.is_dir():
                issues.append(
                    f"organized container is not a directory: {destination}"
                )
                continue
            try:
                size_bytes, fingerprint = snapshot_directory(destination)
            except (OSError, SnapshotLimitError):
                issues.append(
                    f"organized directory cannot be snapshotted: {destination}"
                )
                continue
            if size_bytes != operation.size_bytes:
                issues.append(f"organized directory size changed: {destination}")
            if fingerprint != operation.tree_fingerprint:
                issues.append(
                    f"organized directory contents changed: {destination}"
                )
        elif not destination.is_file():
            issues.append(
                f"organized entry is not a regular file: {destination}"
            )
            continue
        try:
            relative_destination = destination.relative_to(root)
        except ValueError:
            issues.append(f"organized file is outside the root: {destination}")
            continue
        current = root
        for component in relative_destination.parts[:-1]:
            current = current / component
            if current.is_symlink():
                issues.append(f"organized parent is a symbolic link: {current}")
                break
        if not operation.is_directory:
            stat = destination.stat(follow_symlinks=False)
            if stat.st_size != operation.size_bytes:
                issues.append(f"organized file size changed: {destination}")
            if stat.st_mtime_ns != operation.modified_time_ns:
                issues.append(
                    f"organized file modification time changed: {destination}"
                )

        source_key = _path_key(source)
        destination_key = _path_key(destination)
        if source_key in source_keys:
            issues.append(f"duplicate original path: {source}")
        if destination_key in destination_keys:
            issues.append(f"duplicate organized path: {destination}")
        source_keys.add(source_key)
        destination_keys.add(destination_key)
    return issues


def _rollback(
    moved: list[MoveOperation],
    created_directories: list[Path],
) -> list[str]:
    """Best-effort rollback of moves and empty directories created here."""

    issues: list[str] = []

    for operation in reversed(moved):
        try:
            if operation.source.exists() or operation.source.is_symlink():
                issues.append(
                    f"source path is occupied during rollback: "
                    f"{operation.source}"
                )
                continue
            _move_without_overwrite(
                operation.destination,
                operation.source,
            )
        except OSError as error:
            issues.append(
                f"could not restore {operation.source.name}: {error}"
            )

    for directory in reversed(created_directories):
        marker = directory / CATEGORY_MARKER_NAME
        try:
            if marker.is_file() and not marker.is_symlink():
                marker.unlink()
        except OSError:
            pass
        try:
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            if directory.exists():
                issues.append(
                    f"could not remove created directory: {directory}"
                )

    return issues


def _move_without_overwrite(source: Path, destination: Path) -> None:
    """Move one file or directory without knowingly overwriting a destination."""

    if source.is_dir() and not source.is_symlink():
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"destination already exists: {destination}")
        os.rename(source, destination)
        return

    os.link(
        source,
        destination,
        src_dir_fd=None,
        dst_dir_fd=None,
        follow_symlinks=False,
    )
    try:
        source.unlink()
    except OSError as unlink_error:
        try:
            destination.unlink()
        except OSError as cleanup_error:
            raise OSError(
                f"could not remove source or undo destination link: "
                f"{unlink_error}; cleanup failed: {cleanup_error}"
            ) from unlink_error
        raise


def _path_key(path: Path) -> str:
    """Return a normalized key for duplicate-path checks."""

    return unicodedata.normalize("NFKC", str(path)).casefold()
