import csv
import pickle
import shutil
from typing import Sequence
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.data.loaders import load_declarations
from src.data.models import Regulation
from src.preprocessing.text import normalize_declaration, normalize_regulation
from src.retrieval.regulation_matching import RegulationMatcher
from src.retrieval.regulation_retriever import RegulationRetriever
from src.retrieval.tnved_retriever import TNVEDRetriever
from src.reranking.reranker import RegulationReranker
from src.utils.logger import get_logger, setup_logger
from src.utils.get_device import get_device


logger = get_logger(__name__)

# Пути
ROOT_DIR = Path(__file__).resolve().parent


DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
INDEX_DIR = ROOT_DIR / "indexes"
OUTPUT_PATH = ROOT_DIR / "out" / "predictions.csv"
OUTPUT_PATH_TEXTS = ROOT_DIR / "out" / "texts.csv"

DECLARATIONS_PATH = DATA_DIR / "declarations.jsonl"
REGULATIONS_PATH = DATA_DIR / "regulations.jsonl"

TNVED_INDEX_DIR = INDEX_DIR / "tnved"
REGULATION_INDEX_DIR = INDEX_DIR / "regulations"
REGULATION_DATA_PATH = REGULATION_INDEX_DIR / "regulations.pkl"

BGE_MODEL_PATH = MODELS_DIR / "bge_m3"
E5_MODEL_PATH = MODELS_DIR / "e5"
RERANKER_MODEL_PATH = MODELS_DIR / "reranker_model"


def main() -> None:
    setup_logger()

    logger.info("Начало предсказания")
    device = get_device()

    # 1. Загрузка деклараций
    logger.info(f"Загрузка деклараций: {DECLARATIONS_PATH}")
    declarations = load_declarations(DECLARATIONS_PATH)
    logger.info(f"Загружено деклараций: {len(declarations)}")
    if not declarations:
        raise ValueError("Список деклараций пуст")

    with REGULATION_DATA_PATH.open("rb") as file:
        regulations: Sequence[Regulation] = pickle.load(file)

    # 2. Нормализация деклараций
    logger.info("Нормализация деклараций")
    declarations = [normalize_declaration(declaration) for declaration in declarations]

    # 3. Загрузка двух embedding-моделей
    logger.info(f"Загрузка embedding-модели bge-m3: {BGE_MODEL_PATH}")

    embedding_model_bge = SentenceTransformer(
        str(BGE_MODEL_PATH),
        device=device,
        local_files_only=True,
    )
    logger.info(f"Модель регуляций загружена на устройство: "
                f"{embedding_model_bge.device}")

    logger.info(f"Загрузка embedding-модели e5-small: {E5_MODEL_PATH}")

    embedding_model_e5 = SentenceTransformer(
        str(E5_MODEL_PATH),
        device=device,
        local_files_only=True,
    )

    logger.info(f"Модель ТН ВЭД загружена на устройство: {embedding_model_e5.device}")

    # 4. Загрузка retrievers
    regulation_retriever_bge = RegulationRetriever(
        model=embedding_model_bge,
        index_dir=REGULATION_INDEX_DIR,
    )
    regulation_retriever_e5 = RegulationRetriever(
        model=embedding_model_e5,
        index_dir=REGULATION_INDEX_DIR,
        e5=True
    )
    tnved_retriever = TNVEDRetriever(
        model=embedding_model_e5,
        index_dir=TNVED_INDEX_DIR,
    )

    # 5. Загрузка matcher
    matcher = RegulationMatcher(
        regulation_retriever_bge,
        regulation_retriever_e5,
        tnved_retriever
    )
    logger.info("Инициализация RegulationMatcher")

    # 6. Загрузка reranker
    # reranker = RegulationReranker(
    #     model_path=RERANKER_MODEL_PATH,
    #     regulations=regulations,
    #     embedding_model=embedding_model_e5,
    #     max_length=1024,
    #     batch_size=16,
    #     device=device,
    # )

    # 7. Предсказание
    # получение эмбеддингов деклараций из двух моделей
    texts = [d.product_description for d in declarations]
    declaration_embeddings = regulation_retriever_bge.encode_batch(texts, model=embedding_model_bge)
    declaration_embeddings_e5 = regulation_retriever_e5.encode_batch(texts, model=embedding_model_e5)

    # получение контекста ТН ВЭД и enriched queries
    enriched_queries = []
    logger.info("start ТН ВЭД retrieval")
    tnved_results_batch = []

    for declaration, embedding in zip(declarations, declaration_embeddings_e5):
        tnved_results = tnved_retriever.retrieve(
            declaration.product_description,
            embedding,
            top_k=10,
        )
        tnved_results_batch.append(tnved_results)
        tnved_context = tnved_retriever.build_context([node for node, _ in tnved_results])
        enriched_queries.append(
            matcher.build_enriched_query(
                declaration.product_description,
                tnved_context,
            )
        )
    logger.info("end ТН ВЭД retrieval")
    logger.info("start encoding enriched queries")
    # получение эмбеддингов расширенного контекста с ТН ВЭД
    enriched_embeddings = regulation_retriever_e5.encode_batch(
        enriched_queries,
        model=embedding_model_e5,
        batch_size=8,
    )
    logger.info("end of encoding enriched queries")
    rows: list[dict] = []
    texts_out = []

    for i, (declaration, embedding) in enumerate(zip(declarations, declaration_embeddings)):
        logger.info(
            f"Обработка декларации {i}/{len(declarations)}: "
            f"{declaration.declaration_id}"
        )
        text = declaration.product_description
        enriched_embedding = enriched_embeddings[i]
        enriched_query = enriched_queries[i]
        matches = matcher.match(
            text,
            embedding,
            enriched_query,
            enriched_embedding,
            matcher_top_k=50,
            regulation_retrieval_top_k=50,
            tnved_retrieval_top_k=10,
            direct_weight=0.5,
            tnved_weight=0.5,
        )

        # logger.info("start reranking")
        # reranked = reranker.rerank(
        #     declaration_text=text,
        #     query_embedding=declaration_embeddings_e5[i],
        #     candidates=[match.regulation for match in matches[:40]],
        #     tnved_nodes=[node for node, _ in tnved_results_batch[i]],
        #     top_k=10
        # )
        # logger.info("end reranking")

        reg_texts = []
        for rank, match in enumerate(matches[:10], 1):
            rows.append(
                {
                    "declaration_id": declaration.declaration_id,
                    "rank": rank,
                    "regulation_id": (
                        match.regulation.regulation_id
                    ),
                    "score": match.score,
                }
            )
            reg_texts.append(f"{rank}\n"
                             f"РЕГУЛЯЦИЯ №{match.regulation.regulation_id}:\n"
                             f"ТЕКСТ РЕГУЛЯЦИИ: {match.regulation.text}")
        sep = "/n/n" + ("-"*10) + "/n/n"
        texts_out.append(
            {
                "declaration_text": f"ДЕКЛАРАЦИЯ №{declaration.declaration_id}:\n"
                                    f"ТЕКСТ ДЕКЛАРАЦИИ: {declaration.product_description}",
                "regulation_texts": sep.join(reg_texts)
            }
        )

    logger.info(f"Предсказания построены: {len(rows)} строк")

    # 8. Сохранение
    if OUTPUT_PATH.exists():
        shutil.rmtree(OUTPUT_PATH)

    OUTPUT_PATH.mkdir(parents=True)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as file:
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

    with OUTPUT_PATH_TEXTS.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "declaration_text",
                "regulation_texts"
            ]
        )
        writer.writeheader()
        writer.writerows(texts_out)

    logger.info(f"Предсказания сохранены: {OUTPUT_PATH}")
    logger.info("Предсказание завершено")


if __name__ == "__main__":
    main()
