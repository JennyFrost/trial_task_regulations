from __future__ import annotations

import logging
import time
from typing import Iterable
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.models import TNVEDNode
from src.utils.logger import get_logger, setup_logger

setup_logger(
    log_level=logging.INFO,
    log_file="logs/app.log",
)
logger = get_logger(__name__)


class TNVEDRetriever:
    """Dense retrieval по иерархии ТН ВЭД."""

    def __init__(
            self,
            nodes: list[TNVEDNode],
            model: SentenceTransformer,
            batch_size: int = 64
    ) -> None:
        self.model = model
        self.nodes = nodes
        self.batch_size = batch_size

        if not self.nodes:
            raise ValueError("Список узлов ТН ВЭД пуст")

        # для embedding используем полный путь по иерархии
        texts = [node.full_path_text for node in self.nodes if node.leaf_text]
        print(f"Embedding device: {self.model.device}")

        # test_texts = texts[:1000]
        #
        # start = time.perf_counter()
        #
        # embeddings = self.model.encode(
        #     test_texts,
        #     batch_size=48,
        #     normalize_embeddings=True,
        #     convert_to_numpy=True,
        #     show_progress_bar=True,
        # )
        #
        # torch.cuda.synchronize()
        #
        # logger.info(
        #     "1000 texts encoded in %.2f sec",
        #     time.perf_counter() - start,
        # )

        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        logger.info(f"Эмбеддинги узлов ТН ВЭД сформированы с помощью модели {self.model}")

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[TNVEDNode, float]]:
        """Возвращает top-k наиболее релевантных узлов ТН ВЭД и cosine similarity."""

        if not query.strip():
            return []
        if top_k <= 0:
            return []
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # для нормализованных эмбеддингов dot product = cosine similarity
        scores = self.embeddings @ query_embedding
        top_k = min(top_k, len(scores))

        # находим top-k без полной сортировки массива
        indices = np.argpartition(scores, -top_k)[-top_k:]

        # сортируем найденные результаты по убыванию score
        indices = indices[np.argsort(scores[indices])[::-1]]
        return [(self.nodes[i], float(scores[i])) for i in indices]

    @staticmethod
    def build_context(
        results: Iterable[tuple[TNVEDNode, float]],
        include_scores: bool = False,
    ) -> str:
        """Преобразует результаты retrieval в текстовый контекст."""

        lines = []
        for node, score in results:
            if include_scores:
                lines.append(
                    f"[{score:.4f}] {node.full_path_text}"
                )
            else:
                lines.append(node.full_path_text)
        return "\n".join(lines)