"""Cross-encoder reranker for improving retrieval quality.

Uses a cross-encoder model (e.g. BGE-reranker or OpenScholar Reranker) to
rescore query–passage pairs after initial bi-encoder retrieval. Cross-encoders
jointly attend to both the query and passage, giving more accurate relevance
scores than dot-product similarity.

Requires ``sentence-transformers`` (already in local_requirements.txt).
"""

from __future__ import annotations

from typing import Sequence

from .types import RetrievalMatch


class CrossEncoderReranker:
    """Rerank retrieval matches using a cross-encoder model.

    Example::

        reranker = CrossEncoderReranker("BAAI/bge-reranker-v2-m3")
        reranked = reranker.rerank(matches, query="How much energy to train an LLM?")

    The reranker is loaded lazily on first use so it doesn't block pipeline
    construction when not needed.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
                Good options:
                - ``"BAAI/bge-reranker-v2-m3"`` (small, fast, multilingual)
                - ``"BAAI/bge-reranker-large"`` (better quality, ~1.3 GB)
                - ``"OpenSciLM/OpenScholar_Reranker"`` (science-tuned, ~1.2 GB)
            device: PyTorch device (``"cuda"``, ``"cpu"``, or ``None`` for auto).
            batch_size: Batch size for cross-encoder inference.
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # lazy loaded

    def _load_model(self):
        """Load the cross-encoder model on first use."""
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(
            self._model_name,
            device=self._device,
        )

    def rerank(
        self,
        matches: list[RetrievalMatch],
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalMatch]:
        """Rescore and reorder matches by cross-encoder relevance.

        Args:
            matches: Retrieval matches from bi-encoder search.
            query: The original user question.
            top_k: If set, return only the top-k results after reranking.

        Returns:
            Reranked list of :class:`RetrievalMatch` (scores updated to
            cross-encoder scores).
        """
        if not matches:
            return matches

        self._load_model()

        # Build (query, passage) pairs
        pairs = [(query, m.node.text) for m in matches]

        # Score all pairs
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )

        # Pair scores with matches and sort descending
        scored = sorted(
            zip(scores, matches),
            key=lambda x: x[0],
            reverse=True,
        )

        # Update match scores and collect results
        result: list[RetrievalMatch] = []
        for score, match in scored:
            # Create a new RetrievalMatch with the cross-encoder score
            result.append(RetrievalMatch(
                node=match.node,
                score=float(score),
            ))

        if top_k is not None:
            result = result[:top_k]

        return result

    def rerank_texts(
        self,
        texts: Sequence[str],
        query: str,
    ) -> list[tuple[int, float]]:
        """Score raw text passages against a query.

        Returns list of ``(original_index, score)`` sorted by score descending.
        Useful for reranking Semantic Scholar abstracts or other external text.
        """
        if not texts:
            return []

        self._load_model()
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        indexed = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(i, float(s)) for i, s in indexed]
