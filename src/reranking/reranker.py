from typing import Sequence
from sentence_transformers import CrossEncoder

from src.data import Regulation, RerankedRegulation


class RegulationReranker:
    """Переранжирование регуляций с помощью CrossEncoder"""

    def __init__(
        self, model_name: str = "qilowoq/bge-reranker-v2-m3-en-ru",
        max_length: int = 512, batch_size: int = 16, device: str | None = None
    ) -> None:
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.batch_size = batch_size

    def rerank(
        self, declaration_text: str, candidates: Sequence[Regulation],
        top_k: int | None = None
    ) -> list[RerankedRegulation]:
        """Оценивает пары declaration–regulation и сортирует кандидатов"""

        if not candidates:
            return []

        pairs = [(declaration_text, regulation.text) for regulation in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)

        results = [
            RerankedRegulation(regulation, float(score))
            for regulation, score in zip(candidates, scores)
        ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k] if top_k is not None else results
