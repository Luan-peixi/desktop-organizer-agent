"""Bounded read-only inventory for conversational organization requests."""

from __future__ import annotations

import os
from collections.abc import Collection
from pathlib import Path

from desktop_agent.content_extractor import IGNORED_CONTAINER_DIRECTORIES
from desktop_agent.instruction_models import InstructionEntry
from desktop_agent.scanner import is_protected_agent_path

MAX_INSTRUCTION_DEPTH = 3
MAX_INSTRUCTION_ENTRIES = 500


def scan_instruction_entries(
    root: Path,
    selected_names: Collection[str],
) -> dict[str, InstructionEntry]:
    """Inventory selected entries and nested descendants in display order."""

    resolved_root = root.resolve(strict=True)
    selected = set(selected_names)
    paths: list[Path] = []
    for top_level in sorted(resolved_root.iterdir(), key=_sort_key):
        if top_level.name not in selected or not _is_safe_entry(top_level):
            continue
        if is_protected_agent_path(top_level):
            continue
        paths.append(top_level)
        if top_level.is_dir():
            paths.extend(_walk_container(top_level))
        if len(paths) >= MAX_INSTRUCTION_ENTRIES:
            break

    return _build_entries(resolved_root, paths[:MAX_INSTRUCTION_ENTRIES])


def scan_instruction_paths(
    root: Path,
    scope_paths: Collection[Path],
) -> dict[str, InstructionEntry]:
    """Inventory existing organized results for a follow-up adjustment."""

    resolved_root = root.resolve(strict=True)
    paths: list[Path] = []
    seen: set[str] = set()
    for raw_path in sorted(scope_paths, key=_sort_key):
        try:
            path = raw_path.resolve(strict=True)
            relative = path.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        if not relative.parts or len(relative.parts) > 4:
            continue
        if not _is_safe_entry(path):
            continue
        key = str(path).casefold()
        if key not in seen:
            paths.append(path)
            seen.add(key)
        if path.is_dir():
            for descendant in _walk_container(path):
                try:
                    descendant_relative = descendant.relative_to(resolved_root)
                except ValueError:
                    continue
                if len(descendant_relative.parts) > 4:
                    continue
                descendant_key = str(descendant).casefold()
                if descendant_key not in seen:
                    paths.append(descendant)
                    seen.add(descendant_key)
                if len(paths) >= MAX_INSTRUCTION_ENTRIES:
                    break
        if len(paths) >= MAX_INSTRUCTION_ENTRIES:
            break
    return _build_entries(resolved_root, paths[:MAX_INSTRUCTION_ENTRIES])


def _build_entries(
    resolved_root: Path,
    paths: list[Path],
) -> dict[str, InstructionEntry]:
    """Create stable prompt IDs and sibling display orders for safe paths."""

    entries: dict[str, InstructionEntry] = {}
    sibling_orders: dict[str | None, int] = {}
    for position, path in enumerate(paths, start=1):
        relative = path.relative_to(resolved_root)
        parent_relative = (
            relative.parent.as_posix() if len(relative.parts) > 1 else None
        )
        sibling_orders[parent_relative] = sibling_orders.get(parent_relative, 0) + 1
        stat = path.stat(follow_symlinks=False)
        is_directory = path.is_dir()
        size_bytes = stat.st_size
        entry_id = f"entry_{position:04d}"
        entries[entry_id] = InstructionEntry(
            entry_id=entry_id,
            path=path,
            relative_path=relative.as_posix(),
            name=path.name,
            parent_relative_path=parent_relative,
            display_order=sibling_orders[parent_relative],
            depth=len(relative.parts),
            is_directory=is_directory,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            modified_time_ns=stat.st_mtime_ns,
            tree_fingerprint=None,
        )
    return entries


def _walk_container(container: Path) -> list[Path]:
    """Return descendants in deterministic parent-before-child display order."""

    descendants: list[Path] = []
    for current_root, directory_names, file_names in os.walk(
        container,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        depth_from_container = len(current.relative_to(container).parts)
        directory_names[:] = sorted(
            (
                name
                for name in directory_names
                if _is_safe_entry(current / name)
                and name.casefold() not in IGNORED_CONTAINER_DIRECTORIES
                and depth_from_container < MAX_INSTRUCTION_DEPTH - 1
            ),
            key=str.casefold,
        )
        children = sorted(
            [current / name for name in directory_names]
            + [
                current / name
                for name in file_names
                if _is_safe_entry(current / name)
            ],
            key=_sort_key,
        )
        for child in children:
            if len(child.relative_to(container).parts) <= MAX_INSTRUCTION_DEPTH:
                descendants.append(child)
                if len(descendants) >= MAX_INSTRUCTION_ENTRIES:
                    return descendants
    return descendants


def _is_safe_entry(path: Path) -> bool:
    """Exclude hidden entries, links, special files, and the Agent itself."""

    return (
        not path.name.startswith(".")
        and not path.is_symlink()
        and (path.is_file() or path.is_dir())
        and not is_protected_agent_path(path)
    )


def _sort_key(path: Path) -> str:
    return path.name.casefold()
