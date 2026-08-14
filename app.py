"""Local Streamlit interface for the desktop organizer agent."""

from __future__ import annotations

import base64
import os
import uuid
from pathlib import Path
from typing import Any

import streamlit as st
from langgraph.types import Command
from openai import (
    APIConnectionError,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)

from desktop_agent.course_knowledge import load_course_database
from desktop_agent.executor import (
    ExecutionError,
    ExecutionPreviewError,
    get_last_execution_move_count,
    undo_last_execution,
)
from desktop_agent.graph import build_organizer_graph
from desktop_agent.image_analyzer import OpenAIImageAnalyzer
from desktop_agent.instruction_agent import OpenAIInstructionPlanner
from desktop_agent.instruction_executor import build_instruction_preview
from desktop_agent.instruction_models import InstructionAction
from desktop_agent.instruction_scanner import (
    scan_instruction_entries,
    scan_instruction_paths,
)
from desktop_agent.model_client import ModelPlanError, OpenAIPlanGenerator
from desktop_agent.models import AssignmentStatus
from desktop_agent.plan_validator import PlanValidationError
from desktop_agent.platform_paths import find_desktop_directory
from desktop_agent.scanner import list_selectable_entries


def main() -> None:
    """Render the local web application."""

    st.set_page_config(
        page_title="智能桌面整理 Agent",
        page_icon="🗂️",
        layout="wide",
    )
    _apply_styles()

    st.title("智能桌面整理 Agent")
    st.caption("由 OpenAI、LangGraph 与本地安全执行器驱动")
    undo_message = st.session_state.pop("undo_message", None)
    if undo_message:
        st.success(undo_message)
    adjustment_message = st.session_state.pop("adjustment_message", None)
    if adjustment_message:
        st.success(adjustment_message)

    with st.sidebar:
        st.subheader("运行设置")
        default_directory = str(find_desktop_directory())
        directory_text = st.text_input(
            "待整理目录",
            value=default_directory,
            help=(
                "默认自动识别当前电脑用户的桌面；"
                "也可手动输入其他目录。"
            ),
        )
        selection_directory = str(Path(directory_text).expanduser())
        if st.session_state.get("selection_directory") != selection_directory:
            st.session_state.selection_directory = selection_directory
            st.session_state.selected_desktop_entries = []
            st.session_state.pop("instruction_workflow", None)
        selectable_names = _load_selectable_names(directory_text)
        selection_mode = st.radio(
            "整理范围",
            options=("手动选择", "全选"),
            horizontal=True,
            help="未选中的桌面项目不会被扫描、发送给模型或移动。",
        )
        if selection_mode == "全选":
            selected_names = selectable_names
            st.caption(f"已选择全部 {len(selected_names)} 个顶层项目。")
        else:
            selected_names = st.multiselect(
                "选择需要整理的项目",
                options=selectable_names,
                placeholder="点击选择文件、文件夹或压缩包",
                key="selected_desktop_entries",
            )
        read_content = st.toggle(
            "读取文档与图片内容",
            value=True,
            help="提取有限正文并分析图片内容，发送给 OpenAI 辅助分类。",
            key="content_read_enabled",
        )

        if read_content:
            st.warning(
                "正文摘录和图片会发送给模型，请勿用于未经授权的敏感文件。"
            )
            st.caption("本次分析将读取受限正文与图片内容。")

        content_required = (
            not read_content
            and _selection_contains_containers(directory_text, selected_names)
        )
        if content_required:
            st.error(
                "所选内容包含文件夹或 ZIP。仅看外层名称容易漏分，"
                "请开启“读取文档与图片内容”后再分析。"
            )

        if os.getenv("OPENAI_API_KEY"):
            st.success("OpenAI API Key 已配置")
        else:
            st.error("未检测到 OPENAI_API_KEY")

        st.info(
            f"本地课程知识库：{len(load_course_database())} 门课程；"
            "库外课程仍由 GPT 动态识别。"
        )

        analyze = st.button(
            "开始智能分析",
            type="primary",
            use_container_width=True,
            disabled=not selected_names or content_required,
        )

        st.divider()
        st.subheader("对话式整理")
        instruction_text = st.text_area(
            "告诉 Agent 你想怎么整理",
            placeholder="",
            height=110,
            key="instruction_input",
        )
        send_instruction = st.button(
            "发送给偏好 Agent",
            use_container_width=True,
            disabled=(
                not selected_names
                or not instruction_text.strip()
                or not os.getenv("OPENAI_API_KEY")
            ),
        )

        undo_count = _get_undo_count(directory_text)
        if undo_count is not None:
            st.caption(f"检测到上次整理记录：可撤回 {undo_count} 个文件。")
            if st.button(
                "撤回上次整理",
                use_container_width=True,
            ):
                _undo_from_ui(directory_text)

        st.divider()
        st.caption("安全边界")
        st.markdown(
            "- 只整理顶层项目，容器内部只读检查\n"
            "- 自动排除当前 Agent 项目及包含它的目录\n"
            "- 跳过隐藏文件与软链接\n"
            "- 移动前必须人工确认\n"
            "- 禁止覆盖和路径越界"
        )

        st.divider()
        st.subheader("界面显示")
        agent_card_scale = st.slider(
            "Agent 展示缩放",
            min_value=55,
            max_value=100,
            value=70,
            step=5,
            help="同步缩放右侧六张 Agent 卡片，方便完整截图。",
        )

    _apply_agent_card_scale(agent_card_scale)

    if analyze:
        _start_analysis(directory_text, read_content, selected_names)

    if send_instruction:
        _handle_instruction_message(
            directory_text,
            selected_names,
            instruction_text.strip(),
        )

    instruction_workflow = st.session_state.get("instruction_workflow")
    if instruction_workflow is not None:
        _render_instruction_workflow(instruction_workflow)

    adjustment_workflow = st.session_state.get("adjustment_workflow")
    if adjustment_workflow is not None:
        _render_adjustment_workflow(adjustment_workflow)

    adjustment_context = st.session_state.get("adjustment_context")
    if adjustment_context is not None:
        _render_adjustment_box(adjustment_context)

    workflow = st.session_state.get("workflow")
    if workflow is None:
        _render_welcome()
        return

    _render_workflow(workflow)


def _handle_instruction_message(
    directory_text: str,
    selected_names: list[str],
    message: str,
) -> None:
    """Interpret one user turn against a fresh, read-only inventory."""

    directory = Path(directory_text).expanduser()
    previous = st.session_state.get("instruction_workflow")
    messages = list(previous.get("messages", [])) if previous else []
    messages.append({"role": "user", "content": message})
    try:
        with st.spinner("偏好 Agent 正在理解指令并检查文件清单……"):
            entries = scan_instruction_entries(directory, selected_names)
            plan = OpenAIInstructionPlanner().generate_plan(messages, entries)
            preview = build_instruction_preview(plan, entries, directory)
    except Exception as error:
        _show_error(error)
        return

    if plan.clarification_required:
        assert plan.clarification_question is not None
        messages.append(
            {"role": "assistant", "content": plan.clarification_question}
        )
    else:
        messages.append({"role": "assistant", "content": plan.summary})
    st.session_state.instruction_workflow = {
        "directory": directory,
        "selected_names": list(selected_names),
        "messages": messages,
        "entries": entries,
        "plan": plan,
        "preview": preview,
        "phase": "clarification" if plan.clarification_required else "review",
    }
    st.rerun()


def _render_instruction_workflow(workflow: dict[str, Any]) -> None:
    """Render the preference conversation and its exact operation preview."""

    st.divider()
    st.header("对话式整理")
    for message in workflow["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    plan = workflow["plan"]
    if plan.clarification_required:
        st.info("请在左侧输入框回答上面的问题，Agent 会结合对话重新规划。")
        return

    entries = workflow["entries"]
    rows = []
    for operation in plan.operations:
        entry = entries[operation.entry_id]
        rows.append(
            {
                "操作": "移动" if operation.action is InstructionAction.MOVE else "保持原位",
                "条目": entry.relative_path,
                "目标": (
                    "/".join(operation.destination_categories)
                    if operation.destination_categories
                    else "原位置"
                ),
                "理由": operation.reason,
            }
        )
    st.subheader("偏好 Agent 的理解")
    st.write(plan.summary)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("该指令不会产生文件移动。")

    preview = workflow["preview"]
    if workflow["phase"] == "review":
        st.warning("以上仅为预演。未明确列出的文件保持原位。")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button(
                "确认执行对话计划",
                type="primary",
                use_container_width=True,
                disabled=not preview.operations,
            ):
                _execute_instruction_workflow(workflow)
        with cancel:
            if st.button("取消对话计划", use_container_width=True):
                st.session_state.pop("instruction_workflow", None)
                st.rerun()
    elif workflow["phase"] == "executed":
        st.success(
            f"执行完成：已移动 {len(workflow['result'].moved)} 个项目，"
            "可使用左侧“撤回上次整理”恢复。"
        )


def _execute_instruction_workflow(workflow: dict[str, Any]) -> None:
    """Execute an already previewed conversational plan after confirmation."""

    from desktop_agent.executor import execute_preview

    try:
        with st.spinner("正在重新验证并执行对话计划……"):
            result = execute_preview(workflow["preview"], workflow["directory"])
    except Exception as error:
        _show_error(error)
        return
    workflow["result"] = result
    workflow["phase"] = "executed"
    st.session_state.instruction_workflow = workflow
    _start_adjustment_context(workflow["directory"], result)
    st.rerun()


def _start_adjustment_context(directory: Path, result: Any) -> None:
    """Remember the currently organized results available for refinement."""

    st.session_state.adjustment_context = {
        "directory": directory,
        "scope_paths": [operation.destination for operation in result.moved],
        "messages": [],
    }
    st.session_state.pop("adjustment_workflow", None)


def _render_adjustment_box(context: dict[str, Any]) -> None:
    """Offer a new natural-language plan over the just-organized results."""

    st.subheader("继续调整整理结果")
    st.caption("如果查看桌面后不满意，可以在这里说明需要如何修改。")
    adjustment = st.text_area(
        "输入调整要求",
        placeholder="",
        height=100,
        key="adjustment_input",
    )
    if st.button(
        "让 Agent 重新规划",
        use_container_width=True,
        disabled=(not adjustment.strip() or not os.getenv("OPENAI_API_KEY")),
    ):
        _handle_adjustment_message(context, adjustment.strip())


def _handle_adjustment_message(
    context: dict[str, Any],
    message: str,
) -> None:
    """Build a second safe plan over the results of the previous execution."""

    messages = list(context.get("messages", []))
    messages.append({"role": "user", "content": message})
    try:
        with st.spinner("调整 Agent 正在重新检查整理结果……"):
            entries = scan_instruction_paths(
                context["directory"],
                context["scope_paths"],
            )
            plan = OpenAIInstructionPlanner().generate_plan(messages, entries)
            preview = build_instruction_preview(
                plan,
                entries,
                context["directory"],
            )
    except Exception as error:
        _show_error(error)
        return

    response = (
        plan.clarification_question
        if plan.clarification_required
        else plan.summary
    )
    assert response is not None
    messages.append({"role": "assistant", "content": response})
    st.session_state.adjustment_workflow = {
        "directory": context["directory"],
        "scope_paths": list(context["scope_paths"]),
        "messages": messages,
        "entries": entries,
        "plan": plan,
        "preview": preview,
        "phase": "clarification" if plan.clarification_required else "review",
    }
    context["messages"] = messages
    st.session_state.adjustment_context = context
    st.rerun()


def _render_adjustment_workflow(workflow: dict[str, Any]) -> None:
    """Show follow-up conversation, clarification, and exact second preview."""

    st.divider()
    st.header("整理结果的二次调整")
    for message in workflow["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    plan = workflow["plan"]
    if plan.clarification_required:
        st.info("请在下方的调整对话框回答 Agent 的问题。")
        return

    rows = []
    for operation in plan.operations:
        entry = workflow["entries"][operation.entry_id]
        rows.append(
            {
                "操作": "移动" if operation.action is InstructionAction.MOVE else "保持原位",
                "当前位置": entry.relative_path,
                "调整到": (
                    "/".join(operation.destination_categories)
                    if operation.destination_categories
                    else "原位置"
                ),
                "理由": operation.reason,
            }
        )
    st.subheader("二次调整预演")
    st.write(plan.summary)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("这次调整不会移动文件。")
    confirm, cancel = st.columns(2)
    with confirm:
        if st.button(
            "确认执行二次调整",
            type="primary",
            use_container_width=True,
            disabled=not workflow["preview"].operations,
        ):
            _execute_adjustment_workflow(workflow)
    with cancel:
        if st.button("取消本次调整", use_container_width=True):
            st.session_state.pop("adjustment_workflow", None)
            st.rerun()


def _execute_adjustment_workflow(workflow: dict[str, Any]) -> None:
    """Execute a confirmed refinement and keep its new locations adjustable."""

    from desktop_agent.executor import execute_preview

    try:
        with st.spinner("正在重新验证并执行二次调整……"):
            result = execute_preview(workflow["preview"], workflow["directory"])
    except Exception as error:
        _show_error(error)
        return

    moved_sources = {operation.source for operation in result.moved}
    new_scope = [
        path
        for path in workflow["scope_paths"]
        if path not in moved_sources and path.exists()
    ]
    new_scope.extend(operation.destination for operation in result.moved)
    st.session_state.adjustment_context = {
        "directory": workflow["directory"],
        "scope_paths": new_scope,
        "messages": [],
    }
    st.session_state.pop("adjustment_workflow", None)
    st.session_state.adjustment_message = (
        f"二次调整完成：已移动 {len(result.moved)} 个项目。"
    )
    st.rerun()


def _start_analysis(
    directory_text: str,
    read_content: bool,
    selected_names: list[str],
) -> None:
    """Start a fresh graph thread and run it up to human review."""

    if not os.getenv("OPENAI_API_KEY"):
        st.error("请先在启动 Streamlit 的终端配置 OPENAI_API_KEY。")
        return

    directory = Path(directory_text).expanduser()
    generator = OpenAIPlanGenerator()
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
        with st.spinner("正在扫描文件并生成动态整理计划……"):
            state = graph.invoke(
                {
                    "directory": directory,
                    "read_content": read_content,
                    "selected_names": selected_names,
                },
                config=config,
            )
    except Exception as error:
        _show_error(error)
        return

    phase = "review" if "__interrupt__" in state else "complete"
    st.session_state.workflow = {
        "graph": graph,
        "config": config,
        "state": state,
        "directory": directory,
        "read_content": read_content,
        "selected_names": selected_names,
        "phase": phase,
    }


def _load_selectable_names(directory_text: str) -> list[str]:
    """List selectable items while keeping invalid paths non-fatal."""

    try:
        return list_selectable_entries(Path(directory_text).expanduser())
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return []


def _selection_contains_containers(
    directory_text: str,
    selected_names: list[str],
) -> bool:
    """Detect selections that need internal evidence for reliable planning."""

    root = Path(directory_text).expanduser()
    for name in selected_names:
        candidate = root / name
        try:
            if candidate.is_dir() or candidate.suffix.casefold() == ".zip":
                return True
        except OSError:
            continue
    return False


def _render_welcome() -> None:
    """Explain the multi-agent workflow before the first run."""

    st.info("在左侧选择一个模拟目录，然后点击“开始智能分析”。")
    st.subheader("六个 Agent 分工协作")
    steps = [
        ("1", "扫描 Agent", "识别顶层文件、文件夹与压缩包", "rgba(184, 30, 39, 0.89)", "center 18%"),
        ("2", "内容 Agent", "可选理解正文、图片和容器内容", "rgba(33, 81, 160, 0.89)", "center 18%"),
        ("3", "课程 Agent", "匹配跨专业课程知识", "rgba(63, 132, 85, 0.86)", "center 8%"),
        ("4", "规划 Agent", "GPT 动态生成分类方案", "rgba(43, 160, 161, 0.86)", "center 14%"),
        ("5", "安全 Agent", "验证置信度和目标路径", "rgba(239, 241, 233, 0.81)", "center 13%"),
        ("6", "执行 Agent", "人工批准后移动文件", "rgba(41, 126, 72, 0.89)", "center 14%"),
    ]
    cards = []
    for index, (number, title, description, tint, position) in enumerate(
        steps,
        start=1,
    ):
        image_uri = _agent_card_image_uri(index)
        cards.append(
            "<article class='step-card' "
            f"style=\"--agent-image: url('{image_uri}'); "
            f"--agent-tint: {tint}; --agent-position: {position}\">"
            "<div class='step-card-content'>"
            f"<span class='step-number'>{number}</span>"
            "<div class='step-copy'>"
            f"<div class='step-title'>{title}</div>"
            f"<div class='step-description'>{description}</div>"
            "</div>"
            "</div></article>"
        )
    st.markdown(
        "<div class='step-card-grid'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _agent_card_image_uri(index: int) -> str:
    """Load one bundled card image as a browser-safe data URI."""

    image_path = Path(__file__).parent / "assets" / "agent_cards" / f"agent-{index}.jpg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _apply_agent_card_scale(scale_percent: int) -> None:
    """Scale all six showcase cards while preserving their 3:4 ratio."""

    scale = scale_percent / 100
    st.markdown(
        f"""
        <style>
        .step-card-grid {{
            --agent-card-scale: {scale:.2f};
            width: {scale_percent}%;
            margin-inline: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow(workflow: dict[str, Any]) -> None:
    """Display graph state and offer resume actions when interrupted."""

    state = workflow["state"]
    files_by_id = state.get("files_by_id", {})
    if not files_by_id:
        st.info("目录中没有发现可处理的文件。")
        return

    plan = state["plan"]
    preview = state["preview"]
    directory = workflow["directory"].resolve()

    metric_columns = st.columns(4)
    metric_columns[0].metric("扫描文件", len(files_by_id))
    metric_columns[1].metric("动态类别", len(plan.categories))
    metric_columns[2].metric("可自动整理", len(preview.operations))
    metric_columns[3].metric("需要确认", len(preview.skipped))

    st.subheader("多 Agent 协作轨迹")
    for index, event in enumerate(state.get("agent_trace", []), start=1):
        st.markdown(f"**{index}.** {event}")

    st.subheader("整理方案")
    st.write(plan.overview)

    category_rows = [
        {
            "类别": category.name,
            "上级类别": _parent_name(category.parent_id, plan.categories),
            "说明": category.description,
        }
        for category in plan.categories
    ]
    st.dataframe(category_rows, use_container_width=True, hide_index=True)

    categories_by_id = {
        category.id: category.name for category in plan.categories
    }
    assignment_rows = []
    assignments_by_id = {
        assignment.file_id: assignment for assignment in plan.assignments
    }
    for file_id, file in files_by_id.items():
        assignment = assignments_by_id[file_id]
        assignment_rows.append(
            {
                "项目": file.name,
                "类型": "文件夹" if file.is_directory else "文件",
                "建议类别": categories_by_id.get(
                    assignment.category_id,
                    "未确定",
                ),
                "置信度": f"{assignment.confidence:.0%}",
                "状态": (
                    "已分类"
                    if assignment.status is AssignmentStatus.CLASSIFIED
                    else "需要确认"
                ),
                "内容状态": file.content_status.value,
                "学术材料": file.academic_material_hint or "—",
                "课程提示": file.course_hint or "—",
                "分类理由": assignment.reason,
            }
        )

    st.subheader("文件识别结果")
    st.dataframe(assignment_rows, use_container_width=True, hide_index=True)

    st.subheader("执行预演")
    if preview.operations:
        operation_rows = [
            {
                "文件": operation.source.name,
                "目标位置": _relative_path(operation.destination, directory),
            }
            for operation in preview.operations
        ]
        st.dataframe(operation_rows, use_container_width=True, hide_index=True)
    else:
        st.info("没有达到自动整理条件的文件。")

    if preview.skipped:
        with st.expander("查看需要人工确认的文件"):
            for skipped in preview.skipped:
                st.write(f"• {skipped.source.name}：{skipped.reason}")

    phase = workflow["phase"]
    if phase == "review":
        st.warning("以上仅为预演。确认前不会创建目录或移动文件。")
        confirm_column, cancel_column = st.columns([1, 1])
        with confirm_column:
            if st.button(
                "确认并执行整理",
                type="primary",
                use_container_width=True,
            ):
                _resume_workflow(workflow, "MOVE")
        with cancel_column:
            if st.button("取消本次整理", use_container_width=True):
                _resume_workflow(workflow, "CANCEL")
    elif phase == "executed":
        result = state["execution_result"]
        st.success(f"整理完成：成功移动 {len(result.moved)} 个文件。")
    elif phase == "canceled":
        st.info("本次整理已取消，没有移动文件。")
    else:
        st.info("没有可自动执行的操作，文件保持不变。")


def _resume_workflow(workflow: dict[str, Any], decision: str) -> None:
    """Resume the paused graph with an approval or cancellation decision."""

    try:
        with st.spinner("正在执行安全整理……"):
            state = workflow["graph"].invoke(
                Command(resume=decision),
                config=workflow["config"],
            )
    except Exception as error:
        _show_error(error)
        return

    workflow["state"] = state
    workflow["phase"] = (
        "executed" if state.get("approved", False) else "canceled"
    )
    st.session_state.workflow = workflow
    if workflow["phase"] == "executed":
        _start_adjustment_context(
            workflow["directory"],
            state["execution_result"],
        )
    st.rerun()


def _get_undo_count(directory_text: str) -> int | None:
    """Return an undo count without disrupting normal page rendering."""

    try:
        return get_last_execution_move_count(
            Path(directory_text).expanduser()
        )
    except (FileNotFoundError, NotADirectoryError, ExecutionError, OSError):
        return None


def _undo_from_ui(directory_text: str) -> None:
    """Restore the latest execution and refresh the page state."""

    directory = Path(directory_text).expanduser()
    try:
        with st.spinner("正在验证并撤回上次整理……"):
            result = undo_last_execution(directory)
    except Exception as error:
        _show_error(error)
        return

    st.session_state.pop("workflow", None)
    st.session_state.pop("instruction_workflow", None)
    st.session_state.pop("adjustment_workflow", None)
    st.session_state.pop("adjustment_context", None)
    st.session_state.undo_message = (
        f"撤回完成：已将 {len(result.restored)} 个文件恢复到原位置。"
    )
    st.rerun()


def _parent_name(parent_id: str | None, categories: list[Any]) -> str:
    """Return a display name for an optional parent category."""

    if parent_id is None:
        return "—"
    return next(
        (category.name for category in categories if category.id == parent_id),
        "—",
    )


def _relative_path(path: Path, root: Path) -> str:
    """Display target paths relative to the selected directory."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return "路径安全检查失败"


def _show_error(error: Exception) -> None:
    """Translate expected backend failures into concise user messages."""

    if isinstance(error, FileNotFoundError):
        st.error("指定目录不存在。")
    elif isinstance(error, NotADirectoryError):
        st.error("指定路径不是目录。")
    elif isinstance(error, PermissionError):
        st.error("没有权限读取该目录。")
    elif isinstance(error, AuthenticationError):
        st.error("OpenAI API 认证失败，请重新配置 API Key。")
    elif isinstance(error, APIConnectionError):
        st.error("无法连接 OpenAI API，请检查网络。")
    elif isinstance(error, RateLimitError):
        st.error("OpenAI API 额度不足或受到限流。")
    elif isinstance(error, PlanValidationError):
        st.error("模型计划未通过安全验证。")
    elif isinstance(error, ExecutionPreviewError):
        st.error("目标路径未通过安全预演检查。")
    elif isinstance(error, ExecutionError):
        st.error(f"文件操作被安全拒绝或执行失败：{error}")
    elif isinstance(error, ModelPlanError):
        st.error("模型没有返回有效的分类计划。")
    elif isinstance(error, OpenAIError):
        st.error("OpenAI 调用失败，请检查模型和账户配置。")
    else:
        st.error(f"运行失败：{type(error).__name__}")


def _apply_styles() -> None:
    """Apply the local application visual system."""

    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(145deg, #f5f7fb 0%, #eef3f8 100%);
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }
        .step-card-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: calc(1rem * var(--agent-card-scale, 1));
        }
        .step-card {
            position: relative;
            isolation: isolate;
            aspect-ratio: 3 / 4;
            min-height: calc(250px * var(--agent-card-scale, 1));
            overflow: hidden;
            border: 1px solid color-mix(in srgb, var(--agent-tint) 72%, white);
            border-radius: calc(18px * var(--agent-card-scale, 1));
            background: #263650;
            box-shadow:
                0 12px 30px rgba(31, 45, 61, 0.15),
                0 0 22px color-mix(in srgb, var(--agent-tint) 34%, transparent);
        }
        .step-card::before {
            content: "";
            position: absolute;
            inset: -3px;
            z-index: -2;
            background-image: var(--agent-image);
            background-position: var(--agent-position);
            background-size: cover;
            filter: blur(1px) saturate(0.9);
            transform: scale(1.015);
        }
        .step-card::after {
            content: "";
            position: absolute;
            inset: 0;
            z-index: -1;
            background:
                radial-gradient(
                    ellipse at center,
                    color-mix(in srgb, var(--agent-tint) 6%, transparent) 0%,
                    color-mix(in srgb, var(--agent-tint) 25%, transparent) 30%,
                    color-mix(in srgb, var(--agent-tint) 67%, transparent) 65%,
                    var(--agent-tint) 100%
                ),
                linear-gradient(
                    180deg,
                    color-mix(in srgb, var(--agent-tint) 41%, transparent) 0%,
                    color-mix(in srgb, var(--agent-tint) 17%, transparent) 38%,
                    rgba(7, 14, 28, 0.32) 67%,
                    color-mix(in srgb, var(--agent-tint) 88%, rgba(7, 14, 28, 0.88)) 100%
                );
        }
        .step-card-content {
            box-sizing: border-box;
            width: 100%;
            height: 100%;
            min-height: calc(250px * var(--agent-card-scale, 1));
            padding: calc(1.15rem * var(--agent-card-scale, 1));
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            text-shadow: 0 2px 9px rgba(0, 0, 0, 0.8);
        }
        .step-card-content > .step-number {
            display: inline-grid;
            width: calc(30px * var(--agent-card-scale, 1));
            height: calc(30px * var(--agent-card-scale, 1));
            flex: 0 0 calc(30px * var(--agent-card-scale, 1));
            place-items: center;
            border-radius: calc(9px * var(--agent-card-scale, 1));
            color: white;
            background: #2563eb;
            font-weight: 700;
            font-size: calc(1rem * var(--agent-card-scale, 1));
        }
        .step-copy {
            width: 100%;
            min-width: 0;
        }
        .step-title {
            margin: 0 0 0.3rem;
            color: white;
            font-size: clamp(
                calc(1.05rem * var(--agent-card-scale, 1)),
                calc(1.55vw * var(--agent-card-scale, 1)),
                calc(1.35rem * var(--agent-card-scale, 1))
            );
            line-height: 1.25;
            font-weight: 700;
            white-space: nowrap;
        }
        .step-description {
            color: rgba(255, 255, 255, 0.92);
            font-size: clamp(
                calc(0.82rem * var(--agent-card-scale, 1)),
                calc(1.05vw * var(--agent-card-scale, 1)),
                calc(0.98rem * var(--agent-card-scale, 1))
            );
            line-height: 1.45;
            overflow-wrap: break-word;
            word-break: normal;
        }
        @media (max-width: 900px) {
            .step-card-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 620px) {
            .step-card-grid {
                grid-template-columns: 1fr;
            }
        }
        [data-testid="stMetric"] {
            border: 1px solid #dce3ec;
            border-radius: 14px;
            background: white;
            padding: 0.8rem 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
