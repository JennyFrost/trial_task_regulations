from dataclasses import dataclass


@dataclass(slots=True)
class Declaration:
    declaration_id: str
    product_description: str = ""
    quantity: int | None = None


@dataclass(slots=True)
class Regulation:
    regulation_id: str
    decree_number: str | None = None
    section: int | None = None
    point: str | None = None
    text: str = ""


@dataclass(slots=True)
class RegulationMatch:
    """Результат сопоставления декларации с регуляцией"""

    regulation: Regulation
    score: float
    direct_score: float = 0.0
    tnved_score: float = 0.0


@dataclass(slots=True)
class RerankedRegulation:
    regulation: Regulation
    score: float


@dataclass(slots=True)
class TNVEDNode:
    code: str
    section: str | None
    group: str | None
    heading: str | None
    subheading: str | None
    leaf_text: str
    full_path_text: str
