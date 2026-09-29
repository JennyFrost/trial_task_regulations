import re
import unicodedata
from difflib import SequenceMatcher

from src.tnved_ranker.data.schemas import Regulation, Declaration

OCR_REPLACEMENTS = {
    "ПРИ БЕДСТВИИ": "ПРИБЕДСТВИИ",
    "С ИСТЕМОЙ": "СИСТЕМОЙ",
    "Д ЛЯ": "ДЛЯ",
    "ПОЛУПРОВОДНИКОВЫМ МАТЕРИАЛОМ": "ПОЛУПРОВОДНИКОВЫМ МАТЕРИАЛОМ",
    "ТЕРИСТОРНО-ДИОДНЫЙ": "ТЕРИСТОРНО-ДИОДНЫЙ",
    "ПСПАСПОРТ": "ПАСПОРТ",
    "ЭЛЕТРОРАЗРЯДНЫХ": "ЭЛЕКТРОРАЗРЯДНЫХ",
    "ЭЛЕКТРОЭРОЗИОННЫЙ": "ЭЛЕКТРОЭРОЗИОННЫЙ",
    "СИСТЕМОЙВОЗБУЖДЕНИЯ": "СИСТЕМОЙ ВОЗБУЖДЕНИЯ",
    "МЕТАЛЛОКЕРАМИЧЕСКОМ": "МЕТАЛЛОКЕРАМИЧЕСКОМ",
    "ТРАНCФОРМАТОР": "ТРАНСФОРМАТОР",
    "ОТНОСИТЕЛ ЬНО": "ОТНОСИТЕЛЬНО",
    "ПОТР ЕБЛЯЕМАЯ": "ПОТРЕБЛЯЕМАЯ",
    "ПРЕОБР АЗОВАНИЯ": "ПРЕОБРАЗОВАНИЯ",
    "ИЗОЛЯТОР АМИ": "ИЗОЛЯТОРАМИ",
    "ДИАМЕТ Р": "ДИАМЕТР",
    "ТО ПЛИВА": "ТОПЛИВА",
    "ПЕРЕП УСКА": "ПЕРЕПУСКА",
    "УДЕРЖ АНИЯ": "УДЕРЖАНИЯ",
    "ОБ МОТКУ": "ОБМОТКУ",
    "КАРО ТАЖА": "КАРОТАЖА",
    "СОДЕ РЖАЩИЙСЯ": "СОДЕРЖАЩИЙСЯ",
    "ИЗ БЫТОЧНОЕ": "ИЗБЫТОЧНОЕ",
    "ДЛЯУДОБСТВА": "ДЛЯ УДОБСТВА",
    "В КАЧЕСТВЕКОМПЛЕКТУЮЩЕГО": "В КАЧЕСТВЕ КОМПЛЕКТУЮЩЕГО",
    "КОТОРАЯС МЕШИВАЕТСЯ": "КОТОРАЯ СМЕШИВАЕТСЯ",
    "П\"РОМЫШЛЕННОСТИ": "ПРОМЫШЛЕННОСТИ",
    "ВЫХ ОДНАЯ": "ВЫХОДНАЯ",
    "ОПАЛЬБОМ": "АЛЬБОМОМ",
    "Т ВЕРДОТЕЛЬНЫХ": "ТВЕРДОТЕЛЬНЫХ",
    "ПРОСТРАНС ТВЕ": "ПРОСТРАНСТВЕ",
    "ИМЕЮЩИЕ.": "ИМЕЮЩИЕ",
    "КОМПЛЕКТАЦИЯКАБЕЛЬ": "КОМПЛЕКТАЦИЯ КАБЕЛЬ",
    "ФУНКЦИИОРГАНАЙЗЕР": "ФУНКЦИИ ОРГАНАЙЗЕР",
    "ПЛАТИНЧАТЫХ": "ПЛАСТИНЧАТЫХ",
    "ПРОВО ДИМАЯ": "ПРОВОДИМАЯ",
    "НАЗНА ЧЕНИЯ": "НАЗНАЧЕНИЯ",
    "АМПЛИТУДОЙС ИГНАЛА": "АМПЛИТУДОЙ СИГНАЛА",
    "ТОНКОГОГ": "ТОНКОГО",
    "ТОЧНОСТЬОБРАБОТКИ": "ТОЧНОСТЬ ОБРАБОТКИ",
    # часто дублирующиеся слова
    "БАЗОВЫЕ БАЗОВЫЕ": "БАЗОВЫЕ",
    "ДЛЯ ДЛЯ": "ДЛЯ",
}

UNIT_RE = re.compile(
    r"(?<=\d)\s+(?="
    r"(?:"
    r"ГБИТ/С|МБИТ/С|КБИТ/С|"
    r"МГЦ|КГЦ|ГЦ|"
    r"КВТ|МВТ|ВТ|"
    r"МБАР|БАР|КПА|ПА|КВА|"
    r"Л/МИН|Л/С|ОБ/МИН|"
    r"ММ/МИН|М/МИН|М/С|М3/ЧАС|КУБ\.М/Ч|М3/Ч|"
    r"ММ3|ММ2|СМ3|СМ2|КГС/СМ2|"
    r"МКМ|ММ|СМ|КГ|ТБ|ГБ|МБ|КБ|"
    r"А\*Ч|В|М|ГРАД"
    r")"
    r"(?![А-ЯA-Z0-9]))"
)
REGULATION_HEADER_RE = re.compile(
    r"^РАЗДЕЛ\s+(?P<section>\d+)\s*,\s*"
    r"(?P<point>\d+(?:\.\d+)+)\.?\s*",
    flags=re.IGNORECASE,
)


def _normalize_unicode(text: str) -> str:
    """Приводит Unicode к единому виду и удаляет невидимые символы."""

    text = unicodedata.normalize("NFKC", text)

    # неразрывный и zero-width пробелы
    text = text.replace("\u00A0", " ")
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

    # управляющие символы, кроме обычных пробельных
    text = "".join(char for char in text if char in "\n\r\t"
                   or not unicodedata.category(char).startswith("C"))

    # единый регистр и написание ё
    text = text.upper().replace("Ё", "Е")
    return text


def _remove_garbage_symbols(text: str) -> str:
    """Удаляет очевидный OCR и форматный мусор."""

    # например: ^|, |&, $&
    text = re.sub(r"[|^$]+", " ", text)

    # в корпусе используется преимущественно как сломанный разделитель
    text = re.sub(r"\s*&\s*", " ", text)
    return text


def _apply_ocr_corrections(text: str) -> str:
    """Исправляет только известные высоконадежные OCR-ошибки."""

    for old, new in OCR_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def _normalize_formatting(text: str) -> str:
    """Нормализует тире, числа, размеры, единицы и пробелы."""

    # разные варианты тире
    text = re.sub(r"[–—−‒]", "-", text)

    # знак умножения в размерах:
    # 435 x 145 x 285 -> 435X145X285
    # только между числами, чтобы не затрагивать модели
    text = re.sub(
        r"(?<=\d)\s*[×XХxх]\s*(?=\d)",
        "X",
        text,
    )

    # остальные знаки × -- математическое умножение
    text = text.replace("×", "*")

    # 10 000 -> 10000
    # не затрагивает конструкции вроде "1 5/8"
    text = re.sub(r"(?<!\d)(\d{1,3})\s+(\d{3})(?!\d)",r"\1\2", text)

    # 0 - 650 -> 0-650
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)","-", text)

    # 230 / 400 -> 230/400
    text = re.sub(r"\s*/\s*","/", text)

    # убираем пробелы вокруг пунктуации
    text = re.sub(r"\s+([,.;:)])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)

    # убираем пустые скобки, появившиеся после удаления идентификаторов
    text = re.sub(r"\(\s*\)", "", text)

    # не допускаем несколько знаков препинания подряд
    text = re.sub(r"([,.;:])\s*[,. ;:]+", r"\1 ", text)

    # двоеточие и точка с запятой
    text = re.sub(r"\s*:\s*", ": ", text)
    text = re.sub(r"\s*;\s*", "; ", text)

    # склеиваем число с известной единицей:
    # 1400 Л/МИН -> 1400Л/МИН
    # 0-650 ГЦ   -> 0-650ГЦ
    #
    # произвольные число + буквы не трогаем, чтобы не повредить обозначения моделей
    text = UNIT_RE.sub("", text)

    # финальная нормализация пробелов
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _remove_adjacent_duplicate_words(text: str) -> str:
    """Удаляет только непосредственно повторяющиеся слова."""

    words = text.split()
    if not words:
        return ""
    result = [words[0]]
    for word in words[1:]:
        if word != result[-1]:
            result.append(word)
    return " ".join(result)


def normalize_base(text: str) -> str:
    """
    Общая безопасная нормализация деклараций и регуляций.
    Не выполняет лемматизацию, удаление стоп-слов или агрессивное исправление OCR.
    """
    if not text:
        return ""

    text = str(text)
    text = _normalize_unicode(text)
    text = _remove_garbage_symbols(text)
    text = _apply_ocr_corrections(text)
    text = _normalize_formatting(text)
    text = _remove_adjacent_duplicate_words(text)

    return text


def deduplicate_text(
    text: str,
    min_block_words: int = 8,
    similarity_threshold: float = 0.92,
) -> str:
    """
    Удаляет повторяющиеся блоки из текста декларации.
    Параметры:
        text:
            Текст G31_1
        min_block_words:
            Минимальное количество слов в блоке, при котором выполняется дедупликация.
            Короткие блоки сохраняются.
        similarity_threshold:
            Порог сходства блоков для SequenceMatcher.
            0.92 означает, что блоки должны быть практически одинаковыми.
    Возвращает:
        Текст без повторяющихся блоков
    """

    if not text:
        return ""

    # разбиваем техническое описание на блоки
    blocks = re.split(r"(?<=[.;])\s+", text)

    result = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # короткие блоки не дедуплицируем, например: "2 ШТ", "380 В", "50 ГЦ"
        if len(block.split()) < min_block_words:
            result.append(block)
            continue

        is_duplicate = False
        for previous in result:
            # сравниваем только с достаточно длинными предыдущими блоками
            if len(previous.split()) < min_block_words:
                continue
            similarity = SequenceMatcher(None, previous, block).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            result.append(block)
    return " ".join(result)


def remove_identifiers(text: str) -> str:
    """Удаляет явно обозначенные идентификаторы экземпляра товара."""

    if not text:
        return ""

    patterns = [
        # СЕРИЙНЫЙ НОМЕР: ABC123
        # СЕРИЙНЫЙ № ABC123
        r"\bСЕРИЙНЫ(?:Й|Е)\s*(?:НОМЕРА?|№)\s*:?\s*[A-ZА-Я0-9./_-]+",
        # СЕР.№0XXXXX9
        r"\bСЕР\.?\s*(?:НОМЕРА?|№)\s*:?\s*[A-ZА-Я0-9./_-]+"
        # ИДЕНТИФИКАЦИОННЫЙ НОМЕР: ABC123
        # ИДЕНТИФИКАЦИОННЫЙ № ABC123
        r"\bИДЕНТИФИКАЦИОННЫ(?:Й|Е)\s+(?:НОМЕРА?|№)\s*:?\s*[A-ZА-Я0-9./_-]+",

        # ЗАВОДСКОЙ НОМЕР / ЗАВОДСКИЕ НОМЕРА
        r"\bЗАВОДСК(?:ОЙ|ИЕ)\s*(?:НОМЕРА?|№)\s*:?\s*[A-ZА-Я0-9./_,-]+",

        # ЗАВ. НОМЕР: ABC123
        r"\bЗАВ\.\s*(?:НОМЕРА?|№)\s*:?((С|ПО)?\s*[A-ZА-Я0-9./_-]+)*",

        # ЧЕРТЕЖНЫЙ НОМЕР: ABC123
        r"\bЧЕРТЕЖНЫ(?:Й|Е)\s*(?:НОМЕРА?|№)\s*:?\s*[A-ZА-Я0-9./_-]+",

        # № ЧЕРТ. ABC123
        r"\b№\s*ЧЕРТ\.\s*[A-ZА-Я0-9./_-]+",
    ]

    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def normalize_declaration(declaration: Declaration) -> Declaration:
    """Нормализует текст товарной декларации"""

    text = declaration.product_description

    if not text:
        return declaration

    text = remove_identifiers(text)
    text = normalize_base(text)
    text = deduplicate_text(text)

    declaration.product_description = text

    return declaration


def normalize_regulation(regulation: Regulation) -> Regulation:
    """
    Нормализует текст регуляции. Сохраняет номера пунктов, технические обозначения,
    числовые характеристики, примечания и ссылки.
    """
    
    text = regulation.text

    # общая нормализация: Unicode, OCR-ошибки, пробелы, знаки пунктуации, единицы и т.д.
    if text:
        text = normalize_base(text)

    if not text:
        return regulation

    # извлекаем "РАЗДЕЛ N, X.Y.Z...", убираем заголовок, сохраняя номер пункта
    # РАЗДЕЛ 1, 6.1.2.3.1. ... --> 6.1.2.3.1. ...
    # сохраняем номер раздела и номер пункта в отдельные поля объекта регуляции
    match = REGULATION_HEADER_RE.match(text)
    if match:
        section = int(match.group("section"))
        point = match.group("point")
        # удаляем только "РАЗДЕЛ N,", номер пункта сохраняем
        text = text[match.start("point"):]
        regulation.section = section
        regulation.point = point

    # унифицируем обозначения примечаний
    text = re.sub(r"\bТЕХНИЧЕСКОЕ\s+ПРИМЕЧАНИЕ\s*[:.]?",
                  "ТЕХНИЧЕСКОЕ ПРИМЕЧАНИЕ:",
                  text)
    text = re.sub(r"\bПРИМЕЧАНИЕ\s*[:.]?","ПРИМЕЧАНИЕ:", text)

    # нормализуем маркеры списков:
    # "а )" -> "а)"
    # "1 )" -> "1)"
    text = re.sub(r"\b([А-Я])\s+\)", r"\1)", text)
    text = re.sub(r"(?<!\d)(\d+)\s+\)", r"\1)", text)

    text = re.sub(r"\s+", " ", text).strip()
    regulation.text = text

    return regulation
