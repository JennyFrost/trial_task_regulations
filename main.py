from __future__ import annotations
import csv
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.data.loaders import load_declarations, load_regulations
from src.preprocessing.text import normalize_declaration, normalize_regulation
from src.preprocessing.tnved_parser import parse_tnved_knowledge
from src.retrieval.regulations import RegulationRetriever
from src.retrieval.tnved import TNVEDRetriever
from src.retrieval.regulation_matching import RegulationMatcher


BASE_DIR = Path(__file__).resolve().parents[1]

DECLARATIONS_PATH = BASE_DIR / "data" / "declarations.jsonl"
REGULATIONS_PATH = BASE_DIR / "data" / "regulations.jsonl"
TNVED_PATH = BASE_DIR / "data" / "tnved_knowledge.txt"

OUTPUT_PATH = BASE_DIR / "out" / "predictions.csv"

MODEL_NAME = "BAAI/bge-m3"


def main() -> None:
    # ---------------------------------------------------------
    # 1. Загрузка
    # ---------------------------------------------------------

    declarations = load_declarations(
        DECLARATIONS_PATH,
    )

    regulations = load_regulations(
        REGULATIONS_PATH,
    )
    print(1)
    print(declarations[0])
    print(regulations[0])
    print("=*100")

    # ---------------------------------------------------------
    # 2. Нормализация
    # ---------------------------------------------------------

    declarations = [
        normalize_declaration(declaration)
        for declaration in declarations
    ]
    regulations = [
        normalize_regulation(regulation)
        for regulation in regulations
    ]
    print(2)
    print(declarations[0])
    print(regulations[0])
    print("=*100")
    # ---------------------------------------------------------
    # 3. Загрузка базы ТН ВЭД
    # ---------------------------------------------------------

    tnved_nodes = parse_tnved_knowledge(
        TNVED_PATH,
    )
    print(3)
    print(tnved_nodes[0])
    # ---------------------------------------------------------
    # 4. Общая embedding-модель
    # ---------------------------------------------------------

    model = SentenceTransformer(
        MODEL_NAME,
    )

    # ---------------------------------------------------------
    # 5. Retrieval
    # ---------------------------------------------------------

    regulation_retriever = RegulationRetriever(
        regulations,
        model,
    )

    tnved_retriever = TNVEDRetriever(
        tnved_nodes,
        model,
    )

    # ---------------------------------------------------------
    # 6. Matcher
    # ---------------------------------------------------------

    matcher = RegulationMatcher(
        regulation_retriever,
        tnved_retriever,
    )

    # ---------------------------------------------------------
    # 7. Prediction
    # ---------------------------------------------------------

    rows: list[dict] = []

    for declaration in declarations:
        matches = matcher.match(
            declaration.product_description,
            top_k=10,
            candidate_k=20,
            tnved_top_k=5,
            direct_weight=0.7,
            tnved_weight=0.3,
        )

        for rank, match in enumerate(matches, start=1):
            rows.append(
                {
                    "declaration_id": declaration.declaration_id,
                    "rank": rank,
                    "regulation_id": match.regulation.regulation_id,
                    "score": match.score,
                }
            )
    print(7)
    print(rows[0])
    # ---------------------------------------------------------
    # 8. Сохранение
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "declaration_id",
                "rank",
                "regulation_id",
                "score",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()