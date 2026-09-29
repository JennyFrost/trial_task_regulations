from __future__ import annotations
import re
import numpy as np
from typing import Iterable
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.data.models import Regulation


class RegulationRetriever:
    """
    Гибридный retrieval регуляций: dense + BM25 --> RRF --> список ранжированных регуляций
    """

    def __init__(self, regulations: Iterable[Regulation], model: SentenceTransformer) -> None:
        self.regulations = list(regulations)
        self.model = model

        if not self.regulations:
            raise ValueError("Список регуляций пуст")

        # текст для retrieval: только содержание регуляции
        self.texts = [
            regulation.text
            for regulation in self.regulations
        ]

        # эмбеддинги для векторного поиска
        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # BM25 индекс
        tokenized_texts = [self._tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized_texts)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Токенизует русский технический текст для BM25"""

        return re.findall(
            r"[a-zа-яё]+(?:[-'][a-zа-яё]+)*|\d+(?:[.,]\d+)?(?:[-/]\d+)*",
            text.lower())

    def dense_retrieve(self, query: str, top_k: int = 20) \
            -> list[tuple[Regulation, float]]:
        """Возвращает top-k регуляций по dense similarity"""

        if not query.strip() or top_k <= 0:
            return []

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # так как embeddings нормализованы, dot product = cosine similarity
        scores = self.embeddings @ query_embedding
        top_k = min(top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        return [(self.regulations[i], float(scores[i])) for i in indices]

    def bm25_retrieve(self, query: str, top_k: int = 20) \
            -> list[tuple[Regulation, float]]:
        """Возвращает top-k регуляций по BM25"""

        if not query.strip() or top_k <= 0:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)
        top_k = min(top_k, len(scores))
        indices = np.argpartition(scores, -top_k)[-top_k:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        return [(self.regulations[i], float(scores[i])) for i in indices]

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[tuple[Regulation, float]]:
        """Гибридный retrieval: dense retrieval + BM25 --> RRF --> top-k"""

        if not query.strip() or top_k <= 0:
            return []
        if candidate_k <= 0:
            raise ValueError("candidate_k должен быть больше 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k должен быть больше 0")

        dense_results = self.dense_retrieve(query, top_k=candidate_k)
        bm25_results = self.bm25_retrieve(query, top_k=candidate_k)

        return self._rrf_fusion(dense_results, bm25_results, top_k=top_k, rrf_k=rrf_k)

    @staticmethod
    def _rrf_fusion(
        dense_results: list[tuple[Regulation, float]],
        bm25_results: list[tuple[Regulation, float]],
        *,
        top_k: int,
        rrf_k: int = 60,
    ) -> list[tuple[Regulation, float]]:
        """Объединяет два ранжированных списка через Reciprocal Rank Fusion"""

        regulations: dict[str, Regulation] = {}
        rrf_scores: dict[str, float] = {}

        for rank, (regulation, _) in enumerate(dense_results, 1):
            regulation_id = regulation.regulation_id
            regulations[regulation_id] = regulation
            rrf_scores[regulation_id] = (rrf_scores.get(regulation_id, 0.0) + 1.0 /
                                         (rrf_k + rank))

        for rank, (regulation, _) in enumerate(bm25_results, 1):
            regulation_id = regulation.regulation_id
            regulations[regulation_id] = regulation
            rrf_scores[regulation_id] = (rrf_scores.get(regulation_id, 0.0) + 1.0 /
                                         (rrf_k + rank))

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        return [(regulations[regulation_id], rrf_scores[regulation_id])
                for regulation_id in ranked_ids]
