import json
from pathlib import Path

from src.tnved_ranker.data.schemas import Declaration, Regulation


def load_declarations(path: str | Path) -> list[Declaration]:
    path = Path(path)
    declarations = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Невалидный JSON, строка {line_number}") from exc
            declarations.append(
                Declaration(
                        declaration_id=data["declaration_id"],
                        product_description=data.get("G31_1", ""),
                        quantity=data.get("G32"),
                )
            )
    return declarations


def load_regulations(path: str | Path) -> list[Regulation]:
    path = Path(path)
    regulations = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Невалидный JSON, строка {line_number}") from exc
            regulations.append(
                Regulation(
                    regulation_id=data["declaration_id"],
                    decree_number=str(data.get("decree_number", "")),
                    text=data.get("npa", "")
                )
            )
    return regulations
