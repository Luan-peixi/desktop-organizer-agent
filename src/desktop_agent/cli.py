"""Command-line interface for the desktop organizer agent."""

import argparse
import sys
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable, Protocol

from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from langgraph.types import Command

from desktop_agent.executor import (
    ExecutionError,
    ExecutionPreview,
    ExecutionPreviewError,
    build_execution_preview,
)
from desktop_agent.graph import build_organizer_graph
from desktop_agent.image_analyzer import OpenAIImageAnalyzer
from desktop_agent.model_client import ModelPlanError, OpenAIPlanGenerator
from desktop_agent.models import (
    AssignmentStatus,
    FileMetadata,
    OrganizationPlan,
)
from desktop_agent.plan_validator import PlanValidationError, index_files
from desktop_agent.scanner import scan_directory


class PlanGenerator(Protocol):
    """The plan-generation behavior required by the CLI."""

    def generate_plan(
        self,
        files_by_id: Mapping[str, FileMetadata],
    ) -> OrganizationPlan:
        """Generate an organization plan."""


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="desktop-agent",
        description="安全地扫描目录并生成动态文件整理计划。",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    scan_parser = subparsers.add_parser(
        "scan",
        help="只读扫描目录并展示文件元数据",
    )
    scan_parser.add_argument(
        "directory",
        type=Path,
        help="需要扫描的目录路径",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="调用 GPT 生成动态分类计划，但不移动文件",
    )
    plan_parser.add_argument(
        "directory",
        type=Path,
        help="需要生成整理计划的目录路径",
    )

    preview_parser = subparsers.add_parser(
        "preview",
        help="调用 GPT 并预演目标路径，但不创建目录或移动文件",
    )
    preview_parser.add_argument(
        "directory",
        type=Path,
        help="需要预演整理操作的目录路径",
    )

    organize_parser = subparsers.add_parser(
        "organize",
        help="预演后要求明确确认，再安全移动文件",
    )
    organize_parser.add_argument(
        "directory",
        type=Path,
        help="需要整理的目录路径",
    )
    organize_parser.add_argument(
        "--read-content",
        action="store_true",
        help="提取有限文档正文并分析受支持图片，发送给模型",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    plan_generator: PlanGenerator | None = None,
    confirmation_reader: Callable[[str], str] | None = None,
) -> int:
    """Run a CLI command and return a process exit code."""

    args = build_parser().parse_args(argv)

    if args.command == "scan":
        return _run_scan(args.directory)

    if args.command == "organize":
        return _run_graph_organize(
            args.directory,
            plan_generator=plan_generator,
            confirmation_reader=confirmation_reader,
            read_content=args.read_content,
        )

    return _run_plan(
        args.directory,
        plan_generator=plan_generator,
        show_execution_preview=args.command == "preview",
    )


def _scan_for_cli(directory: Path) -> list[FileMetadata] | None:
    """Scan a directory and translate filesystem errors for CLI users."""

    try:
        return scan_directory(directory)
    except FileNotFoundError:
        print(f"错误：指定目录不存在：{directory}", file=sys.stderr)
    except NotADirectoryError:
        print(f"错误：指定路径不是目录：{directory}", file=sys.stderr)
    except PermissionError:
        print(f"错误：没有权限读取该目录：{directory}", file=sys.stderr)

    return None


def _run_scan(directory: Path) -> int:
    """Execute the metadata-only scan command."""

    files = _scan_for_cli(directory)
    if files is None:
        return 2

    print(f"扫描目录：{directory.resolve()}")

    if not files:
        print("没有发现可处理的文件。")
        return 0

    print(f"发现 {len(files)} 个文件：")

    for file in files:
        extension = file.extension or "（无扩展名）"
        print()
        print(f"- {file.name}")
        print(f"  扩展名：{extension}")
        print(f"  大小：{file.size_bytes} B")
        print(f"  修改时间：{file.modified_at.isoformat()}")

    return 0


def _run_plan(
    directory: Path,
    *,
    plan_generator: PlanGenerator | None,
    show_execution_preview: bool = False,
) -> int:
    """Execute the GPT-backed planning command without moving files."""

    files = _scan_for_cli(directory)
    if files is None:
        return 2

    print(f"扫描目录：{directory.resolve()}")

    if not files:
        print("没有发现可处理的文件，不调用模型。")
        return 0

    files_by_id = index_files(files)

    try:
        generator = plan_generator or OpenAIPlanGenerator()
        plan = generator.generate_plan(files_by_id)
        execution_preview = (
            build_execution_preview(plan, files_by_id, directory)
            if show_execution_preview
            else None
        )
    except AuthenticationError:
        print(
            "错误：OpenAI API 认证失败，请检查 OPENAI_API_KEY。",
            file=sys.stderr,
        )
        return 2
    except APIConnectionError:
        print(
            "错误：无法连接 OpenAI API，请检查网络后重试。",
            file=sys.stderr,
        )
        return 2
    except RateLimitError:
        print(
            "错误：OpenAI API 请求受到限流，请稍后重试。",
            file=sys.stderr,
        )
        return 2
    except PlanValidationError as error:
        print(f"错误：模型计划未通过安全验证：\n{error}", file=sys.stderr)
        return 2
    except ExecutionPreviewError as error:
        print(f"错误：无法安全预演整理操作：\n{error}", file=sys.stderr)
        return 2
    except ModelPlanError:
        print("错误：模型没有返回有效的整理计划。", file=sys.stderr)
        return 2
    except OpenAIError:
        print(
            "错误：OpenAI 调用失败，请检查 OPENAI_API_KEY 和模型配置。",
            file=sys.stderr,
        )
        return 2

    _print_plan(plan, files_by_id)
    if execution_preview is not None:
        _print_execution_preview(execution_preview)
    return 0


def _run_graph_organize(
    directory: Path,
    *,
    plan_generator: PlanGenerator | None,
    confirmation_reader: Callable[[str], str] | None,
    read_content: bool,
) -> int:
    """Run the interruptible LangGraph organization workflow."""

    generator = plan_generator or OpenAIPlanGenerator()
    graph = build_organizer_graph(
        generator,
        image_analyzer=OpenAIImageAnalyzer(),
    )
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }

    try:
        if read_content:
            print(
                "已启用内容读取：有限正文和受支持图片将发送给模型。"
            )
        state = graph.invoke(
            {
                "directory": directory,
                "read_content": read_content,
            },
            config=config,
        )
    except FileNotFoundError:
        print(f"错误：指定目录不存在：{directory}", file=sys.stderr)
        return 2
    except NotADirectoryError:
        print(f"错误：指定路径不是目录：{directory}", file=sys.stderr)
        return 2
    except PermissionError:
        print(f"错误：没有权限读取该目录：{directory}", file=sys.stderr)
        return 2
    except AuthenticationError:
        print(
            "错误：OpenAI API 认证失败，请检查 OPENAI_API_KEY。",
            file=sys.stderr,
        )
        return 2
    except APIConnectionError:
        print(
            "错误：无法连接 OpenAI API，请检查网络后重试。",
            file=sys.stderr,
        )
        return 2
    except RateLimitError:
        print(
            "错误：OpenAI API 请求受到限流，请稍后重试。",
            file=sys.stderr,
        )
        return 2
    except PlanValidationError as error:
        print(f"错误：模型计划未通过安全验证：\n{error}", file=sys.stderr)
        return 2
    except ExecutionPreviewError as error:
        print(f"错误：无法安全预演整理操作：\n{error}", file=sys.stderr)
        return 2
    except ModelPlanError:
        print("错误：模型没有返回有效的整理计划。", file=sys.stderr)
        return 2
    except OpenAIError:
        print(
            "错误：OpenAI 调用失败，请检查 OPENAI_API_KEY 和模型配置。",
            file=sys.stderr,
        )
        return 2

    print(f"扫描目录：{directory.resolve()}")
    files_by_id = state["files_by_id"]
    if not files_by_id:
        print("没有发现可处理的文件，不调用模型。")
        return 0

    _print_plan(state["plan"], files_by_id)
    preview = state["preview"]
    _print_execution_preview(preview)

    if not preview.operations:
        print()
        print("没有可执行的移动操作，未更改任何文件。")
        return 0

    reader = confirmation_reader or input
    print()
    try:
        confirmation = reader(
            "请输入 MOVE 确认执行，输入其他内容取消："
        )
    except (EOFError, KeyboardInterrupt):
        print()
        confirmation = "CANCEL"

    try:
        final_state = graph.invoke(
            Command(resume=confirmation),
            config=config,
        )
    except ExecutionError as error:
        print(f"错误：执行整理失败：\n{error}", file=sys.stderr)
        return 2

    if not final_state.get("approved", False):
        print("已取消，未更改任何文件。")
        return 0

    result = final_state["execution_result"]
    print()
    print(f"整理完成：成功移动 {len(result.moved)} 个文件。")
    if preview.skipped:
        print(f"保留 {len(preview.skipped)} 个需要人工确认的文件。")
    return 0


def _print_plan(
    plan: OrganizationPlan,
    files_by_id: Mapping[str, FileMetadata],
) -> None:
    """Display a validated dynamic organization plan."""

    categories_by_id = {
        category.id: category for category in plan.categories
    }
    children_by_parent: dict[str, list[str]] = defaultdict(list)

    for category in plan.categories:
        if category.parent_id is not None:
            children_by_parent[category.parent_id].append(category.id)

    print()
    print(f"整理计划：{plan.overview}")
    print()
    print("动态类别：")

    root_categories = [
        category
        for category in plan.categories
        if category.parent_id is None
    ]
    for root in root_categories:
        print(f"- {root.name}")
        print(f"  说明：{root.description}")

        for child_id in children_by_parent[root.id]:
            child = categories_by_id[child_id]
            print(f"  - {child.name}")
            print(f"    说明：{child.description}")

    print()
    print("文件分配：")

    for assignment in plan.assignments:
        file = files_by_id[assignment.file_id]
        category = (
            categories_by_id.get(assignment.category_id)
            if assignment.category_id is not None
            else None
        )
        category_name = category.name if category is not None else "未确定"
        status_name = (
            "已分类"
            if assignment.status is AssignmentStatus.CLASSIFIED
            else "需要确认"
        )

        print()
        print(f"- {file.name}")
        print(f"  建议类别：{category_name}")
        print(f"  置信度：{assignment.confidence:.0%}")
        print(f"  状态：{status_name}")
        print(f"  理由：{assignment.reason}")


def _print_execution_preview(preview: ExecutionPreview) -> None:
    """Display proposed paths without changing the filesystem."""

    print()
    print("执行预演（未创建目录、未移动文件）：")

    if not preview.operations:
        print("没有可自动执行的移动操作。")

    for operation in preview.operations:
        print()
        print(f"- {operation.source.name}")
        print(f"  来源：{operation.source}")
        print(f"  目标：{operation.destination}")

    if preview.skipped:
        print()
        print("跳过并等待人工确认：")
        for skipped in preview.skipped:
            print(f"- {skipped.source.name}：{skipped.reason}")


if __name__ == "__main__":
    raise SystemExit(main())
