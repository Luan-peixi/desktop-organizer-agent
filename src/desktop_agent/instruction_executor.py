"""Safe preview construction for conversational, fine-grained plans."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from desktop_agent.executor import (
    ExecutionPreview,
    ExecutionPreviewError,
    MoveOperation,
    SkippedFile,
)
from desktop_agent.entry_snapshot import SnapshotLimitError, snapshot_directory
from desktop_agent.instruction_agent import _validate_instruction_plan
from desktop_agent.instruction_models import (
    InstructionAction,
    InstructionEntry,
    InstructionPlan,
)
from desktop_agent.scanner import is_protected_agent_path


def build_instruction_preview(
    plan: InstructionPlan,
    entries: Mapping[str, InstructionEntry],
    root: Path,
) -> ExecutionPreview:
    """Translate a validated conversational plan into checked move operations."""

    resolved_root = root.resolve(strict=True)
    _validate_instruction_plan(plan, entries)
    if plan.clarification_required:
        return ExecutionPreview(operations=(), skipped=())

    operations: list[MoveOperation] = []
    skipped: list[SkippedFile] = []
    issues: list[str] = []
    destination_keys: set[str] = set()

    for operation in plan.operations:
        entry = entries[operation.entry_id]
        source = entry.path
        if operation.action is InstructionAction.KEEP:
            skipped.append(
                SkippedFile(
                    file_id=entry.entry_id,
                    source=source,
                    reason="用户指定保持原位",
                )
            )
            continue

        issue = _validate_entry_snapshot(entry, resolved_root)
        if issue is not None:
            issues.append(f"{entry.entry_id}: {issue}")
            continue
        destination = resolved_root.joinpath(
            *operation.destination_categories,
            source.name,
        )
        if destination.exists() or destination.is_symlink():
            issues.append(f"{entry.entry_id}: 目标已存在，禁止覆盖 {destination}")
            continue
        destination_key = str(destination).casefold()
        if destination_key in destination_keys:
            issues.append(f"{entry.entry_id}: 生成了重复目标 {destination}")
            continue
        destination_keys.add(destination_key)
        size_bytes = entry.size_bytes
        tree_fingerprint = entry.tree_fingerprint
        if entry.is_directory:
            try:
                size_bytes, tree_fingerprint = snapshot_directory(source)
            except (OSError, SnapshotLimitError):
                issues.append(f"{entry.entry_id}: 文件夹无法建立安全快照")
                continue
        operations.append(
            MoveOperation(
                file_id=entry.entry_id,
                source=source,
                destination=destination,
                size_bytes=size_bytes,
                modified_time_ns=entry.modified_time_ns,
                is_directory=entry.is_directory,
                tree_fingerprint=tree_fingerprint,
            )
        )

    if issues:
        raise ExecutionPreviewError(issues)
    return ExecutionPreview(operations=tuple(operations), skipped=tuple(skipped))


def _validate_entry_snapshot(entry: InstructionEntry, root: Path) -> str | None:
    """Ensure a model-selected entry is unchanged and still in the allowed root."""

    source = entry.path
    if is_protected_agent_path(source):
        return "Agent 项目受保护"
    if source.is_symlink() or not source.exists():
        return "条目不存在或已变为软链接"
    try:
        relative = source.relative_to(root)
    except ValueError:
        return "条目超出待整理目录"
    if not relative.parts or len(relative.parts) > 4:
        return "条目超出允许的三层内部扫描深度"
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        if current.is_symlink():
            return "条目的上级目录是软链接"
    stat = source.stat(follow_symlinks=False)
    if stat.st_mtime_ns != entry.modified_time_ns:
        return "条目在对话规划后已被修改"
    if entry.is_directory:
        if not source.is_dir():
            return "条目不再是文件夹"
    elif not source.is_file() or stat.st_size != entry.size_bytes:
        return "文件大小在对话规划后已变化"
    return None
