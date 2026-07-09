"""Day 4 — semantic retrieval from Weaviate.

The Retriever accepts a raw user question, normalizes it, embeds it with
the SAME embedder that indexed the chunks, asks Weaviate for the top-k
nearest chunks and returns them as `RetrievedChunk` objects carrying
rank, score and distance — everything later steps need to build a
grounded, cited answer and to detect weak context.

Two modes:
  * vector (default) — pure semantic nearest-neighbour search;
    score = certainty (0..1), distance = cosine distance.
  * hybrid — Weaviate blends BM25 keyword relevance with vector
    similarity (`alpha` controls the mix: 0 = keywords only,
    1 = vectors only); score = fused score, distance = None.
"""

import logging

from embedder import Embedder
from schemas import RetrievedChunk
from weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 4
MAX_QUESTION_CHARS = 1000


def normalize_question(question: str) -> str:
    """Basic user-query processing: collapse whitespace, validate length."""
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("Question is empty")
    if len(normalized) > MAX_QUESTION_CHARS:
        logger.warning(
            "Question is very long (%d chars) — truncating to %d",
            len(normalized), MAX_QUESTION_CHARS,
        )
        normalized = normalized[:MAX_QUESTION_CHARS]
    return normalized


def _to_retrieved_chunks(hits: list[dict], mode: str) -> list[RetrievedChunk]:
    """Map raw Weaviate hits (in returned order) to RetrievedChunk models."""
    results: list[RetrievedChunk] = []
    for rank, hit in enumerate(hits, start=1):
        distance = hit.get("distance")
        if mode == "hybrid":
            score = float(hit.get("score") or 0.0)
        else:
            certainty = hit.get("certainty")
            if certainty is None and distance is not None:
                certainty = 1.0 - distance / 2.0  # cosine distance -> certainty
            score = float(certainty or 0.0)
        results.append(
            RetrievedChunk(
                **hit["properties"], rank=rank, score=score, distance=distance
            )
        )
    return results


class Retriever:
    """Question -> top-k relevant chunks, via the shared store + embedder."""

    def __init__(
        self, store: WeaviateStore, embedder: Embedder, top_k: int = DEFAULT_TOP_K
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        hybrid: bool = False,
        alpha: float = 0.5,
    ) -> list[RetrievedChunk]:
        """Return the top-k chunks most relevant to `question`."""
        question = normalize_question(question)
        top_k = top_k or self.top_k

        vector = self.embedder.embed_query(question)
        if hybrid:
            hits = self.store.search_hybrid(question, vector, top_k=top_k, alpha=alpha)
            results = _to_retrieved_chunks(hits, mode="hybrid")
        else:
            hits = self.store.search_by_vector(vector, top_k=top_k)
            results = _to_retrieved_chunks(hits, mode="vector")

        top = results[0].score if results else float("nan")
        logger.info(
            "Retrieved %d chunks for %r (%s, top score %.3f)",
            len(results), question[:60], "hybrid" if hybrid else "vector", top,
        )
        return results
