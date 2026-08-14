"""LangGraph workflow for safe, human-approved desktop organization."""

from collections.abc import Mapping
from operator import add
from pathlib import Path
from typing import Annotated, Any, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from desktop_agent.content_extractor import extract_contents
from desktop_agent.course_knowledge import (
    enrich_with_course_hints,
    load_course_database,
)
from desktop_agent.executor import (
    ExecutionPreview,
    ExecutionResult,
    build_execution_preview,
    execute_preview,
)
from desktop_agent.models import (
    AssignmentStatus,
    FileMetadata,
    OrganizationPlan,
)
from desktop_agent.plan_validator import index_files
from desktop_agent.scanner import scan_directory


class PlanGenerator(Protocol):
    """The model behavior required by the planning node."""

    def generate_plan(
        self,
        files_by_id: Mapping[str, FileMetadata],
    ) -> OrganizationPlan:
        """Generate and validate a dynamic organization plan."""


class ImageAnalyzer(Protocol):
    """Optional image-understanding behavior used by the content node."""

    def analyze_images(
        self,
        files_by_id: Mapping[str, FileMetadata],
    ) -> dict[str, FileMetadata]:
        """Return metadata enriched with bounded visual descriptions."""


class OrganizerState(TypedDict, total=False):
    """Shared state passed between desktop-organizer graph nodes."""

    directory: Path
    read_content: bool
    selected_names: list[str]
    files_by_id: dict[str, FileMetadata]
    plan: OrganizationPlan
    preview: ExecutionPreview
    decision: str
    approved: bool
    execution_result: ExecutionResult
    agent_trace: Annotated[list[str], add]


def build_organizer_graph(
    plan_generator: PlanGenerator,
    *,
    image_analyzer: ImageAnalyzer | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Build a graph that pauses for approval before moving any file."""

    def scan_node(state: OrganizerState) -> OrganizerState:
        files = scan_directory(
            state["directory"],
            state.get("selected_names"),
        )
        return {
            "files_by_id": index_files(files),
            "agent_trace": [
                f"扫描 Agent｜完成安全扫描，发现 {len(files)} 个顶层文件。"
            ],
        }

    def generate_plan_node(state: OrganizerState) -> OrganizerState:
        plan = plan_generator.generate_plan(state["files_by_id"])
        classified = sum(
            assignment.status is AssignmentStatus.CLASSIFIED
            for assignment in plan.assignments
        )
        return {
            "plan": plan,
            "agent_trace": [
                "规划 Agent｜GPT 动态生成 "
                f"{len(plan.categories)} 个类别，"
                f"{classified} 个文件达到分类条件。"
            ],
        }

    def extract_content_node(state: OrganizerState) -> OrganizerState:
        if not state.get("read_content", False):
            return {
                "agent_trace": [
                    "内容理解 Agent｜用户未启用内容读取，本轮仅使用元数据。"
                ]
            }

        files = extract_contents(state["files_by_id"])
        if image_analyzer is not None:
            files = image_analyzer.analyze_images(files)
        extracted = sum(
            file.content_status.value == "extracted"
            for file in files.values()
        )
        return {
            "files_by_id": files,
            "agent_trace": [
                f"内容理解 Agent｜完成受限内容提取，成功理解 {extracted} 个文件。"
            ],
        }

    def match_courses_node(state: OrganizerState) -> OrganizerState:
        files = enrich_with_course_hints(state["files_by_id"])
        matched = sum(file.course_hint is not None for file in files.values())
        return {
            "files_by_id": files,
            "agent_trace": [
                "课程知识 Agent｜检索 "
                f"{len(load_course_database())} 门课程知识，"
                f"命中 {matched} 个文件。"
            ],
        }

    def build_preview_node(state: OrganizerState) -> OrganizerState:
        preview = build_execution_preview(
            state["plan"],
            state["files_by_id"],
            state["directory"],
        )
        return {
            "preview": preview,
            "agent_trace": [
                "安全审查 Agent｜验证路径、置信度与覆盖风险："
                f"{len(preview.operations)} 个可执行，"
                f"{len(preview.skipped)} 个需人工确认。"
            ],
        }

    def human_review_node(state: OrganizerState) -> OrganizerState:
        preview = state["preview"]
        decision = interrupt(_review_payload(preview))
        decision_text = decision if isinstance(decision, str) else ""
        return {
            "decision": decision_text,
            "approved": decision_text == "MOVE",
            "agent_trace": [
                "人工审核｜"
                + (
                    "用户批准执行。"
                    if decision_text == "MOVE"
                    else "用户取消执行。"
                )
            ],
        }

    def execute_node(state: OrganizerState) -> OrganizerState:
        result = execute_preview(state["preview"], state["directory"])
        return {
            "execution_result": result,
            "agent_trace": [
                f"执行 Agent｜安全移动 {len(result.moved)} 个文件。"
            ],
        }

    builder = StateGraph(OrganizerState)
    builder.add_node("scanner_agent", scan_node)
    builder.add_node("content_agent", extract_content_node)
    builder.add_node("course_agent", match_courses_node)
    builder.add_node("planner_agent", generate_plan_node)
    builder.add_node("safety_agent", build_preview_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("executor_agent", execute_node)

    builder.add_edge(START, "scanner_agent")
    builder.add_conditional_edges(
        "scanner_agent",
        _route_after_scan,
        {
            "has_files": "content_agent",
            "empty": END,
        },
    )
    builder.add_edge("content_agent", "course_agent")
    builder.add_edge("course_agent", "planner_agent")
    builder.add_edge("planner_agent", "safety_agent")
    builder.add_conditional_edges(
        "safety_agent",
        _route_after_preview,
        {
            "needs_review": "human_review",
            "nothing_to_move": END,
        },
    )
    builder.add_conditional_edges(
        "human_review",
        _route_after_review,
        {
            "execute": "executor_agent",
            "cancel": END,
        },
    )
    builder.add_edge("executor_agent", END)

    return builder.compile(
        checkpointer=checkpointer or InMemorySaver(),
        name="desktop-organizer",
    )


def _route_after_scan(state: OrganizerState) -> str:
    """Skip model use when the scanned directory is empty."""

    if not state["files_by_id"]:
        return "empty"
    return "has_files"


def _route_after_review(state: OrganizerState) -> str:
    """Execute only when the human entered the exact approval token."""

    return "execute" if state.get("approved", False) else "cancel"


def _route_after_preview(state: OrganizerState) -> str:
    """Avoid asking for approval when every file requires manual review."""

    return "needs_review" if state["preview"].operations else "nothing_to_move"


def _review_payload(preview: ExecutionPreview) -> dict[str, object]:
    """Build the JSON-serializable information exposed by interrupt()."""

    return {
        "instruction": "输入 MOVE 执行整理，输入其他内容取消。",
        "operations": [
            {
                "file_id": operation.file_id,
                "source": str(operation.source),
                "destination": str(operation.destination),
            }
            for operation in preview.operations
        ],
        "skipped": [
            {
                "file_id": skipped.file_id,
                "source": str(skipped.source),
                "reason": skipped.reason,
            }
            for skipped in preview.skipped
        ],
    }
