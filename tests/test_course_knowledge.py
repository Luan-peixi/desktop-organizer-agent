"""Tests for the local university-course knowledge base."""

from datetime import UTC, datetime
from pathlib import Path

from desktop_agent.course_knowledge import (
    load_course_database,
    match_course,
)
from desktop_agent.models import (
    ContentExtractionStatus,
    FileMetadata,
)


def _file(name: str, content: str | None = None) -> FileMetadata:
    """Build metadata used only by the deterministic matcher."""

    timestamp = datetime(2026, 8, 11, tzinfo=UTC)
    return FileMetadata(
        name=name,
        path=Path("/mock") / name,
        extension=Path(name).suffix.lower(),
        size_bytes=100,
        created_at=timestamp,
        modified_at=timestamp,
        content_status=(
            ContentExtractionStatus.EXTRACTED
            if content is not None
            else ContentExtractionStatus.NOT_REQUESTED
        ),
        content_excerpt=content,
    )


def test_course_database_has_unique_ids_and_core_courses() -> None:
    """Packaged data should remain valid and contain presentation examples."""

    courses = load_course_database()
    ids = [course.id for course in courses]
    names = {course.name for course in courses}
    subjects = {course.subject for course in courses}

    assert len(ids) == len(set(ids))
    assert len(courses) >= 190
    assert "程序设计基础" in names
    assert "计算机组成原理" in names
    assert "习近平新时代中国特色社会主义思想概论" in names
    assert {
        "公共基础类",
        "思想政治类",
        "数学类",
        "自然科学类",
        "计算机类",
        "计算机系统类",
        "软件技术类",
        "人工智能与数据类",
        "法学类",
        "微电子与集成电路类",
        "医学类",
        "机械类",
        "电气与自动化类",
        "土木与建筑类",
        "中国语言文学类",
        "生命科学类",
    }.issubset(subjects)


def test_course_alias_in_filename_produces_a_strong_hint() -> None:
    """A common course abbreviation should resolve to its standard name."""

    match = match_course(_file("计组实验报告.docx"))

    assert match is not None
    assert match.course_name == "计算机组成原理"
    assert match.subject == "计算机类"
    assert match.confidence >= 0.85


def test_distinctive_content_keywords_can_identify_a_course() -> None:
    """Several domain terms should identify a course despite a vague name."""

    match = match_course(
        _file(
            "实验报告.txt",
            "本实验分析 CPU、ALU、寄存器、指令系统和总线结构。",
        )
    )

    assert match is not None
    assert match.course_name == "计算机组成原理"


def test_generic_assignment_language_is_not_forced_into_a_course() -> None:
    """Generic student wording must remain available for GPT judgment."""

    match = match_course(
        _file("实验报告.txt", "本周完成课程实验和课后作业。")
    )

    assert match is None


def test_short_english_alias_does_not_match_inside_an_ordinary_word() -> None:
    """Aliases such as OS must not match an unrelated English filename."""

    match = match_course(_file("position_notes.txt"))

    assert match is None


def test_standalone_short_english_alias_can_identify_a_course() -> None:
    """A separated course abbreviation remains useful after boundary checks."""

    match = match_course(_file("OS 实验报告.txt"))

    assert match is not None
    assert match.course_name == "操作系统"


def test_law_course_filename_resolves_to_standard_course_name() -> None:
    """A law student's abbreviated filename should get a useful hint."""

    match = match_course(_file("民诉案例分析.docx"))

    assert match is not None
    assert match.course_name == "民事诉讼法"
    assert match.subject == "法学类"


def test_microelectronics_content_can_identify_semiconductor_physics() -> None:
    """Distinctive microelectronics terms should identify the course."""

    match = match_course(
        _file(
            "实验报告.txt",
            "实验讨论半导体能带、载流子浓度、费米能级以及 PN结特性。",
        )
    )

    assert match is not None
    assert match.course_name == "半导体物理"
    assert match.subject == "微电子与集成电路类"


def test_full_course_name_outranks_a_generic_alias() -> None:
    """半导体物理 should outrank the generic college-physics alias 物理."""

    match = match_course(_file("半导体物理实验.md"))

    assert match is not None
    assert match.course_name == "半导体物理"


def test_agent_course_folder_resolves_to_artificial_intelligence() -> None:
    """A common agent-assignment folder name should identify AI coursework."""

    match = match_course(
        _file(
            "17-智能体",
            "容器文件清单：\n- 实验报告.docx\n智能体任务规划与工具调用",
        )
    )

    assert match is not None
    assert match.course_name == "人工智能"


def test_english_course_title_resolves_to_artificial_intelligence() -> None:
    """English report headings should map to the standard Chinese course."""

    match = match_course(
        _file("实验报告.pdf", "Artificial Intelligence Agent Experiment")
    )

    assert match is not None
    assert match.course_name == "人工智能"
    assert match.confidence >= 0.85


def test_primary_course_name_outranks_incidental_environment_terms() -> None:
    """Report subject should beat incidental database/OS setup mentions."""

    match = match_course(
        _file(
            "agent大作业_实验报告.pdf",
            "人工智能智能体实验。运行环境使用操作系统与数据库。",
        )
    )

    assert match is not None
    assert match.course_name == "人工智能"
    assert match.confidence >= 0.85
