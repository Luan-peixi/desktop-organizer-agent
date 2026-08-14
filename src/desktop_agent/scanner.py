"""Read-only directory scanning functionality."""

from datetime import UTC, datetime
from collections.abc import Collection
from pathlib import Path

from desktop_agent.entry_snapshot import SnapshotLimitError, snapshot_directory
from desktop_agent.models import FileMetadata

CATEGORY_MARKER_NAME = ".desktop_agent_category"
AGENT_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_protected_agent_path(path: Path) -> bool:
    """Return whether moving this path would move the running Agent itself."""

    try:
        candidate = path.resolve(strict=False)
        project_root = AGENT_PROJECT_ROOT.resolve(strict=True)
    except OSError:
        return False
    return candidate == project_root or project_root.is_relative_to(candidate)


def list_selectable_entries(directory: Path) -> list[str]:
    """List safe top-level choices without reading container contents."""

    root = directory.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")
    return [
        entry.name
        for entry in sorted(root.iterdir(), key=lambda path: path.name.casefold())
        if not entry.name.startswith(".")
        and not entry.is_symlink()
        and (entry.is_file() or entry.is_dir())
        and not is_protected_agent_path(entry)
        and not (
            entry.is_dir()
            and (entry / CATEGORY_MARKER_NAME).is_file()
        )
    ]


def scan_directory(
    directory: Path,
    selected_names: Collection[str] | None = None,
) -> list[FileMetadata]:
    """Return metadata for regular, visible files directly inside a directory."""

    root = directory.resolve(strict=True)

    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    files: list[FileMetadata] = []
    selected = set(selected_names) if selected_names is not None else None

    for entry in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if entry.name.startswith("."):
            continue
        if selected is not None and entry.name not in selected:
            continue

        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            continue
        if is_protected_agent_path(entry):
            continue
        if entry.is_dir() and (entry / CATEGORY_MARKER_NAME).is_file():
            continue

        file_stat = entry.stat(follow_symlinks=False)
        created_timestamp = getattr(file_stat, "st_birthtime", None)
        is_directory = entry.is_dir()
        tree_fingerprint = None
        size_bytes = file_stat.st_size
        if is_directory:
            try:
                size_bytes, tree_fingerprint = snapshot_directory(entry)
            except SnapshotLimitError:
                # Keep the container available for classification, but the
                # executor will refuse automatic movement without a snapshot.
                tree_fingerprint = None

        files.append(
            FileMetadata(
                name=entry.name,
                path=entry,
                extension=entry.suffix.lower(),
                size_bytes=size_bytes,
                created_at=(
                    datetime.fromtimestamp(created_timestamp, tz=UTC)
                    if created_timestamp is not None
                    else None
                ),
                modified_at=datetime.fromtimestamp(file_stat.st_mtime, tz=UTC),
                is_directory=is_directory,
                tree_fingerprint=tree_fingerprint,
            )
        )

    return files
