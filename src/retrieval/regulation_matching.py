# src/retrieval/regulation_matching.py

from __future__ import annotations

from dataclasses import dataclass

from .regulations import Regulation, RegulationRetriever
from .tnved import TNVEDRetriever


@dataclass(slots=True)
class RegulationMatch:
    """Результат сопоставления декларации с регуляцией."""

    regulation: Regulation
    score: float
    direct_score: float = 0.0
    tnved_score: float = 0.0


class RegulationMatcher:
    """
    Сопоставляет декларацию с релевантными регуляциями.
    Используются два retrieval-пути:
    1. Прямой:
       декларация --> RegulationRetriever
    2. С контекстом ТН ВЭД:
       декларация --> TNVEDRetriever --> расширенный запрос --> RegulationRetriever
    Результаты двух путей объединяются через взвешенный RRF.
    """
    def __init__(
        self,
        regulation_retriever: RegulationRetriever,
        tnved_retriever: TNVEDRetriever,
    ) -> None:
        self.regulation_retriever = regulation_retriever
        self.tnved_retriever = tnved_retriever

    def match(
        self,
        declaration_text: str,
        top_k: int = 10,
        candidate_k: int = 20,
        tnved_top_k: int = 5,
        direct_weight: float = 0.7,
        tnved_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[RegulationMatch]:
        """
        Возвращает top-k релевантных регуляций для декларации.
        candidate_k определяет количество кандидатов, получаемых каждым из двух путей retrieval.
        """

        if not declaration_text.strip() or top_k <= 0:
            return []

        if candidate_k <= 0:
            raise ValueError("candidate_k должен быть больше 0")

        if tnved_top_k <= 0:
            raise ValueError("tnved_top_k должен быть больше 0")

        if rrf_k <= 0:
            raise ValueError("rrf_k должен быть больше 0")

        if direct_weight < 0 or tnved_weight < 0:
            raise ValueError("Веса direct_weight и tnved_weight не могут быть отрицательными")

        if direct_weight + tnved_weight == 0:
            raise ValueError("Хотя бы один из весов должен быть больше 0")

        # 1. Прямой retrieval регуляций по декларации
        direct_results = self.regulation_retriever.retrieve(
            declaration_text,
            top_k=candidate_k,
            rrf_k=rrf_k,
        )

        # 2. Retrieval релевантных узлов ТН ВЭД
        tnved_results = self.tnved_retriever.retrieve(declaration_text, top_k=tnved_top_k)

        # 3. Формирование расширенного запроса
        tnved_context = self.tnved_retriever.build_context(tnved_results)
        enriched_query = self._build_enriched_query(declaration_text, tnved_context)

        # 4. Retrieval регуляций по расширенному запросу
        tnved_results = self.regulation_retriever.retrieve(
            enriched_query,
            top_k=candidate_k,
            rrf_k=rrf_k,
        )

        # 5. Объединение двух списков с помощью RRF
        return self._weighted_rrf_fusion(
            direct_results,
            tnved_results,
            direct_weight=direct_weight,
            tnved_weight=tnved_weight,
            top_k=top_k,
            rrf_k=rrf_k,
        )

    @staticmethod
    def _build_enriched_query(declaration_text: str, tnved_context: str) -> str:
        """Добавляет контекст ТН ВЭД к исходному тексту декларации."""

        if not tnved_context:
            return declaration_text

        return (
            f"{declaration_text}\n\n"
            f"Контекст ТН ВЭД:\n"
            f"{tnved_context}"
        )

    @staticmethod
    def _weighted_rrf_fusion(
        direct_results: list[tuple[Regulation, float]],
        tnved_results: list[tuple[Regulation, float]],
        direct_weight: float,
        tnved_weight: float,
        top_k: int,
        rrf_k: int,
    ) -> list[RegulationMatch]:
        """Объединяет результаты двух retrieval-путей через weighted RRF."""

        regulations: dict[str, Regulation] = {}
        scores: dict[str, float] = {}
        direct_scores: dict[str, float] = {}
        tnved_scores: dict[str, float] = {}

        for rank, (regulation, _) in enumerate(direct_results, 1):
            regulation_id = regulation.regulation_id
            regulations[regulation_id] = regulation
            score = direct_weight / (rrf_k + rank)
            scores[regulation_id] = scores.get(regulation_id, 0.0) + score
            direct_scores[regulation_id] = direct_scores.get(regulation_id, 0.0) + score

        for rank, (regulation, _) in enumerate(tnved_results, 1):
            regulation_id = regulation.regulation_id
            regulations[regulation_id] = regulation
            score = tnved_weight / (rrf_k + rank)
            scores[regulation_id] = scores.get(regulation_id, 0.0) + score
            tnved_scores[regulation_id] = tnved_scores.get(regulation_id, 0.0) + score

        ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [RegulationMatch(
                        regulation=regulations[id],
                        score=scores[id],
                        direct_score=direct_scores.get(id, 0.0),
                        tnved_score=tnved_scores.get(id, 0.0))
                for id in ranked_ids]