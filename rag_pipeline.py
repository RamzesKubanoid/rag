"""Day 5/6 — the end-to-end RAG pipeline with weak-context protection.

Day 6 adds two guard layers that run BEFORE the LLM sees anything:

  1. Noise filter — retrieved chunks whose semantic certainty is below
     `min_chunk_score` are dropped from the prompt (obvious junk only).
  2. Weak-context gate — if the BEST certainty is below `weak_threshold`,
     the pipeline refuses immediately with an honest message: no LLM
     call, no tokens spent, no chance to hallucinate.

Certainty is always derived from cosine geometry (1 - distance/2), so
gating works identically for vector and hybrid retrieval. The hybrid
fused score is NEVER thresholded — it is normalized per query, so even
junk queries produce a high-looking top score.

A third, semantic layer lives in the prompt (prompts.py): questions
on-topic enough to pass the gate but unanswered by the fragments must
be refused by the model itself (see the hard negatives in evaluation.py).

Thresholds are embedder-dependent. The defaults suit the offline hashing
embedder; after switching embeddings, run `python main.py --evaluate`
and place the thresholds inside the gap between the good-question and
bad-question confidence ranges it reports.
"""

import logging
import os
import time

from generator import Generator
from prompts import NO_ANSWER_SENTENCE
from retriever import Retriever
from schemas import RAGAnswer, RetrievedChunk, SourceRef

logger = logging.getLogger(__name__)

DEFAULT_WEAK_THRESHOLD = 0.55   # gate: best certainty below this -> refuse pre-LLM
DEFAULT_MIN_CHUNK_SCORE = 0.52  # filter: chunks below this never enter the prompt


def certainty(chunk: RetrievedChunk) -> float:
    """Semantic confidence of one hit on the absolute 0..1 certainty scale.

    Derived from cosine distance when available (present in both vector
    and hybrid modes, since hybrid distance is computed client-side);
    falls back to `score`, which IS the certainty in vector mode.
    """
    if chunk.distance is not None:
        return 1.0 - chunk.distance / 2.0
    return chunk.score


def _sources_from(chunks: list[RetrievedChunk]) -> list[SourceRef]:
    """Source list mirroring the [n] markers of the prompt: marker ==
    1-based position of the chunk in the context actually sent."""
    return [
        SourceRef(
            marker=index,
            chunk_id=chunk.chunk_id,
            source_name=chunk.source_name,
            title=chunk.title,
            score=chunk.score,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


class RAGPipeline:
    """question -> grounded, cited answer — or an honest refusal."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        weak_threshold: float | None = None,
        min_chunk_score: float | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.weak_threshold = (
            float(os.getenv("RETRIEVAL_WEAK_THRESHOLD", DEFAULT_WEAK_THRESHOLD))
            if weak_threshold is None
            else weak_threshold
        )
        self.min_chunk_score = (
            float(os.getenv("RETRIEVAL_MIN_CHUNK_SCORE", DEFAULT_MIN_CHUNK_SCORE))
            if min_chunk_score is None
            else min_chunk_score
        )
        if self.min_chunk_score > self.weak_threshold:
            logger.warning(
                "min_chunk_score (%.2f) > weak_threshold (%.2f) — unusual configuration",
                self.min_chunk_score, self.weak_threshold,
            )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        hybrid: bool = False,
        alpha: float = 0.5,
    ) -> RAGAnswer:
        """The full RAG cycle for one question, with weak-context guards."""
        chunks = self.retriever.retrieve(question, top_k=top_k, hybrid=hybrid, alpha=alpha)

        confidences = [certainty(chunk) for chunk in chunks]
        top_confidence = max(confidences, default=0.0)
        kept = [c for c, conf in zip(chunks, confidences) if conf >= self.min_chunk_score]
        dropped = len(chunks) - len(kept)
        if dropped:
            logger.info(
                "Noise filter: dropped %d/%d chunks below certainty %.2f",
                dropped, len(chunks), self.min_chunk_score,
            )

        if not kept or top_confidence < self.weak_threshold:
            logger.info(
                "Weak context (best certainty %.3f < threshold %.2f) — refusing "
                "WITHOUT calling the LLM", top_confidence, self.weak_threshold,
            )
            refusal = (
                f"{NO_ANSWER_SENTENCE} (No sufficiently relevant fragments were found: "
                f"best retrieval confidence {top_confidence:.2f} is below the "
                f"threshold {self.weak_threshold:.2f}.)"
            )
            return RAGAnswer(
                question=question,
                answer=refusal,
                sources=[],
                chunks=chunks,  # keep the (rejected) hits for transparency/debugging
                model=self.generator.model_name,
                used_retrieval=True,
                weak_context=True,
                retrieval_confidence=top_confidence,
            )

        started = time.perf_counter()
        answer_text = self.generator.answer_with_context(question, kept)
        logger.info(
            "Generated answer with %s in %.2fs (%d context chunks, best certainty %.3f)",
            self.generator.model_name, time.perf_counter() - started,
            len(kept), top_confidence,
        )
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=_sources_from(kept),
            chunks=kept,
            model=self.generator.model_name,
            used_retrieval=True,
            weak_context=False,
            retrieval_confidence=top_confidence,
        )

    def answer_baseline(self, question: str) -> RAGAnswer:
        """No-retrieval answer from the model's own knowledge (comparison only)."""
        started = time.perf_counter()
        answer_text = self.generator.answer_baseline(question)
        logger.info(
            "Generated BASELINE answer with %s in %.2fs (no retrieval)",
            self.generator.model_name, time.perf_counter() - started,
        )
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=[],
            chunks=[],
            model=self.generator.model_name,
            used_retrieval=False,
        )
