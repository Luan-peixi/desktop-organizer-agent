"""Tests for the LangGraph desktop-organizer workflow."""

from pathlib import Path

from langgraph.types import Command

from desktop_agent.graph import build_organizer_graph
from desktop_agent.models import (
    AssignmentStatus,
    CategoryDefinition,
    ContentExtractionStatus,
    FileAssignment,
    OrganizationPlan,
)


class FakePlanGenerator:
    """Return a deterministic plan without making an API request."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_files = None

    def generate_plan(self, files_by_id: object) -> OrganizationPlan:
        """Classify the graph test's only file as project material."""

        self.calls += 1
        self.last_files = files_by_id
        return OrganizationPlan(
            overview="将项目文件整理到工作资料目录。",
            categories=[
                CategoryDefinition(
                    id="work",
                    name="工作资料",
                    description="工作和项目相关文件。",
                    parent_id=None,
                )
            ],
            assignments=[
                FileAssignment(
                    file_id="file_001",
                    category_id="work",
                    confidence=0.98,
                    reason="文件名表明它是项目复盘材料。",
                    status=AssignmentStatus.CLASSIFIED,
                )
            ],
        )


class ReviewOnlyPlanGenerator:
    """Mark the only scanned file as requiring manual review."""

    def __init__(self) -> None:
        self.calls = 0

    def generate_plan(self, files_by_id: object) -> OrganizationPlan:
        """Return a valid plan with no automatically executable operation."""

        self.calls += 1
        return OrganizationPlan(
            overview="文件信息不足，需要人工判断。",
            categories=[
                CategoryDefinition(
                    id="unknown",
                    name="待确认",
                    description="需要人工确认的文件。",
                    parent_id=None,
                )
            ],
            assignments=[
                FileAssignment(
                    file_id="file_001",
                    category_id="unknown",
                    confidence=0.3,
                    reason="当前元数据不足以判断。",
                    status=AssignmentStatus.NEEDS_REVIEW,
                )
            ],
        )


def _config(thread_id: str) -> dict[str, dict[str, str]]:
    """Return the checkpoint configuration for one independent run."""

    return {"configurable": {"thread_id": thread_id}}


def test_graph_pauses_with_serializable_move_preview(
    tmp_path: Path,
) -> None:
    """The graph must pause before any filesystem mutation occurs."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")
    generator = FakePlanGenerator()
    graph = build_organizer_graph(generator)

    paused = graph.invoke(
        {"directory": tmp_path},
        config=_config("pause-test"),
    )

    assert generator.calls == 1
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["instruction"].startswith("输入 MOVE")
    assert payload["operations"] == [
        {
            "file_id": "file_001",
            "source": str(source),
            "destination": str(tmp_path / "工作资料" / source.name),
        }
    ]
    assert source.exists()
    assert not (tmp_path / "工作资料").exists()
    assert [event.split("｜", 1)[0] for event in paused["agent_trace"]] == [
        "扫描 Agent",
        "内容理解 Agent",
        "课程知识 Agent",
        "规划 Agent",
        "安全审查 Agent",
    ]


def test_graph_cancels_without_moving_files(tmp_path: Path) -> None:
    """A non-MOVE resume value should route directly to END."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")
    graph = build_organizer_graph(FakePlanGenerator())
    config = _config("cancel-test")

    graph.invoke({"directory": tmp_path}, config=config)
    result = graph.invoke(Command(resume="CANCEL"), config=config)

    assert result["decision"] == "CANCEL"
    assert result["approved"] is False
    assert "execution_result" not in result
    assert source.exists()
    assert not (tmp_path / "工作资料").exists()


def test_graph_resumes_and_executes_after_move_approval(
    tmp_path: Path,
) -> None:
    """Exact MOVE should resume at the review node and execute the preview."""

    source = tmp_path / "项目复盘.PDF"
    source.write_bytes(b"pdf")
    generator = FakePlanGenerator()
    graph = build_organizer_graph(generator)
    config = _config("execute-test")

    graph.invoke({"directory": tmp_path}, config=config)
    result = graph.invoke(Command(resume="MOVE"), config=config)
    destination = tmp_path / "工作资料" / source.name

    assert generator.calls == 1
    assert result["approved"] is True
    assert len(result["execution_result"].moved) == 1
    assert not source.exists()
    assert destination.read_bytes() == b"pdf"
    assert result["agent_trace"][-2:] == [
        "人工审核｜用户批准执行。",
        "执行 Agent｜安全移动 1 个文件。",
    ]


def test_graph_skips_the_model_for_an_empty_directory(
    tmp_path: Path,
) -> None:
    """An empty scan should finish without API use or an interrupt."""

    generator = FakePlanGenerator()
    graph = build_organizer_graph(generator)

    result = graph.invoke(
        {"directory": tmp_path},
        config=_config("empty-test"),
    )

    assert generator.calls == 0
    assert result["files_by_id"] == {}
    assert "__interrupt__" not in result
    assert "plan" not in result


def test_graph_finishes_without_interrupt_when_every_file_needs_review(
    tmp_path: Path,
) -> None:
    """A graph with no automatic moves should not request MOVE approval."""

    source = tmp_path / "无扩展名文件"
    source.write_text("unknown")
    generator = ReviewOnlyPlanGenerator()
    graph = build_organizer_graph(generator)

    result = graph.invoke(
        {"directory": tmp_path},
        config=_config("review-only-test"),
    )

    assert generator.calls == 1
    assert result["preview"].operations == ()
    assert len(result["preview"].skipped) == 1
    assert "__interrupt__" not in result
    assert source.exists()


def test_graph_extracts_content_before_calling_the_model(
    tmp_path: Path,
) -> None:
    """The opt-in content node should enrich metadata before planning."""

    source = tmp_path / "项目复盘.txt"
    source.write_text("这是火星探测项目的阶段复盘。")
    generator = FakePlanGenerator()
    graph = build_organizer_graph(generator)

    paused = graph.invoke(
        {
            "directory": tmp_path,
            "read_content": True,
        },
        config=_config("content-test"),
    )

    assert "__interrupt__" in paused
    assert generator.last_files is not None
    metadata = generator.last_files["file_001"]
    assert metadata.content_status is ContentExtractionStatus.EXTRACTED
    assert metadata.content_excerpt == "这是火星探测项目的阶段复盘。"


def test_graph_adds_course_hint_before_calling_the_model(
    tmp_path: Path,
) -> None:
    """The local knowledge node should standardize course abbreviations."""

    source = tmp_path / "计组实验报告.txt"
    source.write_text("CPU、ALU、寄存器与指令系统实验。")
    generator = FakePlanGenerator()
    graph = build_organizer_graph(generator)

    paused = graph.invoke(
        {
            "directory": tmp_path,
            "read_content": True,
        },
        config=_config("course-hint-test"),
    )

    assert "__interrupt__" in paused
    metadata = generator.last_files["file_001"]
    assert metadata.course_hint == "计算机组成原理"
    assert metadata.course_subject == "计算机类"
    assert metadata.course_confidence is not None


def test_graph_never_sends_unselected_entry_to_model(tmp_path: Path) -> None:
    """Selection is enforced by the scan node, not merely hidden in the UI."""

    selected = tmp_path / "项目复盘.PDF"
    selected.write_bytes(b"pdf")
    untouched = tmp_path / "私人资料.txt"
    untouched.write_text("private")
    generator = FakePlanGenerator()
    graph = build_organizer_graph(generator)

    paused = graph.invoke(
        {
            "directory": tmp_path,
            "selected_names": [selected.name],
        },
        config=_config("selected-only-test"),
    )

    assert "__interrupt__" in paused
    assert generator.last_files is not None
    assert [file.name for file in generator.last_files.values()] == [
        selected.name
    ]
    assert untouched.exists()
