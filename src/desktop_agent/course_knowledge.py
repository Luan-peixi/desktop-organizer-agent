"""Local university-course knowledge base and deterministic hint matching."""

import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.resources import files

from desktop_agent.models import FileMetadata


@dataclass(frozen=True, slots=True)
class CourseDefinition:
    """One normalized university-course record."""

    id: str
    name: str
    subject: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CourseMatch:
    """A high-confidence local course hint for one file."""

    course_id: str
    course_name: str
    subject: str
    confidence: float
    evidence: tuple[str, ...]


def load_course_database() -> tuple[CourseDefinition, ...]:
    """Load packaged course records from the extensible JSON database."""

    resource = files("desktop_agent").joinpath(
        "data/university_courses.json"
    )
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return tuple(
        CourseDefinition(
            id=record["id"],
            name=record["name"],
            subject=record["subject"],
            aliases=tuple(record["aliases"]),
            keywords=tuple(record["keywords"]),
        )
        for record in payload["courses"]
    )


def enrich_with_course_hints(
    files_by_id: Mapping[str, FileMetadata],
) -> dict[str, FileMetadata]:
    """Return metadata enriched with conservative local course hints."""

    courses = load_course_database()
    enriched: dict[str, FileMetadata] = {}

    for file_id, file in files_by_id.items():
        match = match_course(file, courses)
        if match is None:
            enriched[file_id] = file
            continue
        enriched[file_id] = replace(
            file,
            course_hint=match.course_name,
            course_subject=match.subject,
            course_confidence=match.confidence,
        )

    return enriched


def match_course(
    file: FileMetadata,
    courses: tuple[CourseDefinition, ...] | None = None,
) -> CourseMatch | None:
    """Return a hint only when one course clearly outranks alternatives."""

    course_records = courses or load_course_database()
    filename = file.name
    content = file.content_excerpt or ""
    candidates: list[tuple[int, bool, CourseDefinition, tuple[str, ...]]] = []

    for course in course_records:
        score = 0
        strong_match = False
        evidence: list[str] = []

        # Always prefer a full standard course name over an alias. Otherwise a
        # filename such as "agent大作业" can hide the clearer "人工智能"
        # title found inside its report.
        if _contains(filename, course.name):
            score += 16
            strong_match = True
            evidence.append(course.name)
        elif _contains(content, course.name):
            score += 14
            strong_match = True
            evidence.append(course.name)
        else:
            for label in course.aliases:
                if _contains(filename, label):
                    score += 12
                    strong_match = True
                    evidence.append(label)
                    break
                if _contains(content, label):
                    # A multi-word English course title is substantially more
                    # specific than a short alias such as "Agent" or "OS".
                    is_full_english_title = (
                        label.isascii() and " " in label.strip()
                    )
                    score += 11 if is_full_english_title else 8
                    strong_match = True
                    evidence.append(label)
                    break

        matched_keywords = [
            keyword
            for keyword in course.keywords
            if _contains(f"{filename}\n{content}", keyword)
        ]
        score += 2 * len(matched_keywords)
        evidence.extend(matched_keywords[:3])

        if score >= 6:
            candidates.append(
                (score, strong_match, course, tuple(evidence))
            )

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_is_strong, best_course, evidence = candidates[0]

    if len(candidates) > 1:
        second_score, second_is_strong, _, _ = candidates[1]
        if best_score == second_score:
            return None
        if not best_is_strong and second_score >= best_score - 2:
            return None
        if best_is_strong and second_is_strong and second_score >= best_score - 1:
            return None

    confidence = min(0.99, 0.58 + best_score * 0.025)
    return CourseMatch(
        course_id=best_course.id,
        course_name=best_course.name,
        subject=best_course.subject,
        confidence=confidence,
        evidence=evidence,
    )


def _normalize(value: str) -> str:
    """Normalize case, width, and whitespace for robust local matching."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(normalized.split())


def _contains(text: str, label: str) -> bool:
    """Match labels while protecting short ASCII aliases from substrings."""

    normalized_label = unicodedata.normalize("NFKC", label).casefold().strip()
    normalized_text = unicodedata.normalize("NFKC", text).casefold()
    if not normalized_label:
        return False

    if normalized_label.isascii() and normalized_label.isalnum():
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_label)}(?![a-z0-9])"
        return re.search(pattern, normalized_text) is not None

    return _normalize(normalized_label) in _normalize(normalized_text)
