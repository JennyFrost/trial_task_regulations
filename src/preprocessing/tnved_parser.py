from __future__ import annotations
from pathlib import Path
import re

from src.tnved_ranker.data.schemas import TNVEDNode

_CODE_LENGTHS = {2, 4, 6, 10}
_CODE_RE = re.compile(r"^\d{2}(?:\d{2})?(?:\d{2})?(?:\d{4})?$")
_SECTION_RE = re.compile(r"^[IVXLCDM]+$")
_DESCRIPTION_RE = re.compile(r"\[([^\[\]]+)]\s*$")
_DASH_RE = re.compile(r"^[\s\u00a0]*[-–—−]+\s*")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Убирает лишние пробелы."""
    return _WHITESPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def _extract_description(text: str) -> tuple[str, str | None]:
    """Извлекает описание из квадратных скобок"""

    match = _DESCRIPTION_RE.search(text)

    if not match:
        return text, None

    description = _normalize_text(match.group(1))
    text = text[:match.start()].rstrip()

    return text, description or None


def _strip_hierarchy_markers(text: str) -> tuple[str, int]:
    """Убирает тире и возвращает глубину иерархии"""

    depth = 0
    while match := _DASH_RE.match(text):
        depth += 1
        text = text[match.end():]
    text = _normalize_text(text).rstrip(":").strip()
    return text, depth


def _parse_line(
    line: str,
) -> tuple[str | None, str, str | None, int] | None:
    """Разбирает строку на код, название, описание и глубину"""

    if "|" not in line:
        return None

    code_part, text_part = line.split("|", 1)
    code_part = code_part.strip()
    text_part = _normalize_text(text_part)

    if code_part:
        if len(code_part) not in _CODE_LENGTHS:
            return None
        if not _CODE_RE.fullmatch(code_part):
            return None
        code = code_part
    else:
        code = None
    text_part, description = _extract_description(text_part)
    name, depth = _strip_hierarchy_markers(text_part)
    if not name:
        return None

    return code, name, description, depth


def parse_tnved_knowledge(path: str | Path) -> list[TNVEDNode]:
    """Парсит иерархию ТН ВЭД в плоский список узлов"""

    path = Path(path)
    nodes: list[TNVEDNode] = []
    section: str | None = None
    # текстовый контекст по глубине тире
    context: dict[int, str] = {}

    # последние названия кодированных уровней
    coded_context: dict[int, str] = {}

    with path.open("r", encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            # строка вида "I | ЖИВЫЕ ЖИВОТНЫЕ..."
            if "|" in line:
                left, right = line.split("|", 1)
                left = left.strip()
                if left and _SECTION_RE.fullmatch(left):
                    section = _normalize_text(right)
                    context.clear()
                    coded_context.clear()
                    continue

            parsed = _parse_line(line)
            if parsed is None:
                continue

            code, name, description, depth = parsed

            # строка без кода только обновляет контекст
            if code is None:
                if depth:
                    context[depth] = name

                    for level in list(context):
                        if level > depth:
                            del context[level]
                continue
            code_level = len(code) // 2

            # берём ранее найденные кодированные уровни
            group = coded_context.get(1) or context.get(1)
            heading = coded_context.get(2) or context.get(2)
            subheading = coded_context.get(3) or context.get(3)

            # текущий код заменяет соответствующий уровень
            if code_level == 1:
                group = name
                heading = None
                subheading = None

            elif code_level == 2:
                heading = name
                subheading = None

            elif code_level == 3:
                subheading = name

            leaf_text = description or name

            # сохраняем текущий текстовый контекст
            if depth:
                context[depth] = name
                for level in list(context):
                    if level > depth:
                        del context[level]

            # формируем полный путь, включая некодированные промежуточные узлы
            path_parts = [section, group, heading, subheading]

            for level in sorted(context):
                value = context[level]
                if value not in path_parts:
                    path_parts.append(value)
            if leaf_text not in path_parts:
                path_parts.append(leaf_text)
            full_path_text = " > ".join(
                part for part in path_parts if part
            )

            node = TNVEDNode(
                code=code,
                section=section,
                group=group,
                heading=heading,
                subheading=subheading,
                leaf_text=leaf_text,
                full_path_text=full_path_text,
            )
            nodes.append(node)

            # запоминаем название текущего кодированного уровня
            coded_context[code_level] = name
            for level in list(coded_context):
                if level > code_level:
                    del coded_context[level]

    return nodes

