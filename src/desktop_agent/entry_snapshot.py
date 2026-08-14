"""Deterministic snapshots for bounded top-level directory containers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# A course project can legitimately contain a local virtual environment or
# generated assets.  Keep a firm ceiling, but do not reject ordinary student
# projects merely because they contain more than one thousand entries.
MAX_SNAPSHOT_ENTRIES = 100_000


class SnapshotLimitError(RuntimeError):
    """Raised when a directory is too large for safe automatic movement."""


def snapshot_directory(path: Path) -> tuple[int, str]:
    """Return total regular-file bytes and a stable tree fingerprint."""

    digest = hashlib.sha256()
    total_bytes = 0
    entry_count = 0

    for current_root, directory_names, file_names in os.walk(
        path,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not (current / name).is_symlink()
        )
        for name in [*directory_names, *sorted(file_names)]:
            entry = current / name
            entry_count += 1
            if entry_count > MAX_SNAPSHOT_ENTRIES:
                raise SnapshotLimitError(
                    f"directory exceeds {MAX_SNAPSHOT_ENTRIES} entries"
                )
            stat = entry.stat(follow_symlinks=False)
            relative = entry.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(
                f"|{stat.st_mode}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
            if entry.is_file() and not entry.is_symlink():
                total_bytes += stat.st_size

    return total_bytes, digest.hexdigest()
