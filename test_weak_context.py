"""Self-checks for the Day 6 weak-context guards. No Weaviate or LLM
needed — a fake retriever supplies controlled scores and a counting
generator proves the LLM is (not) called.

Run directly:      python test_weak_context.py
Or with pytest:    pytest test_weak_context.py -q
"""

from prompts import NO_ANSWER_SENTENCE
from rag_pipeline import RAGPipeline, certainty
from schemas import RetrievedChunk


def _chunk(chunk_id: str, rank: int, distance: float, score: float | None = None) -> RetrievedChunk:
    """A retrieved chunk with a given cosine distance (certainty = 1 - d/2)."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_900",
        source_name="synthetic.txt",
        title="Synthetic",
        chunk_index=0,
        text=f"content of {chunk_id}",
        rank=rank,
        score=score if score is not None else 1.0 - distance / 2.0,
        distance=distance,
    )


class _FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question, top_k=None, hybrid=False, alpha=0.5):
        return self.chunks


class _CountingGenerator:
    model_name = "fake-llm"

    def __init__(self):
        self.calls = 0
        self.last_chunks = None

    def answer_with_context(self, question, chunks):
        self.calls += 1
        self.last_chunks = chunks
        return "generated answer [1]"

    def answer_baseline(self, question):
        return "baseline answer"


def test_certainty_derives_from_distance():
    assert abs(certainty(_chunk("c", 1, distance=0.4)) - 0.8) < 1e-9
    no_distance = _chunk("c", 1, distance=0.4).model_copy(update={"distance": None, "score": 0.7})
    assert certainty(no_distance) == 0.7  # falls back to score (vector-mode certainty)


def test_gate_refuses_without_calling_llm():
    generator = _CountingGenerator()
    pipeline = RAGPipeline(_FakeRetriever([_chunk("c1", 1, distance=0.95)]),  # certainty 0.525
                           generator, weak_threshold=0.55, min_chunk_score=0.52)
    result = pipeline.answer("anything")
    assert result.weak_context is True
    assert generator.calls == 0, "the LLM must NOT be called on weak context"
    assert NO_ANSWER_SENTENCE in result.answer
    assert result.sources == [] and abs(result.retrieval_confidence - 0.525) < 1e-9


def test_strong_context_is_answered():
    generator = _CountingGenerator()
    pipeline = RAGPipeline(_FakeRetriever([_chunk("c1", 1, distance=0.4)]),  # certainty 0.8
                           generator, weak_threshold=0.55, min_chunk_score=0.52)
    result = pipeline.answer("anything")
    assert result.weak_context is False and generator.calls == 1
    assert abs(result.retrieval_confidence - 0.8) < 1e-9


def test_noise_filter_drops_junk_and_realigns_markers():
    chunks = [
        _chunk("c_good_1", 1, distance=0.20),  # certainty 0.90
        _chunk("c_junk", 2, distance=1.00),    # certainty 0.50 -> dropped
        _chunk("c_good_2", 3, distance=0.60),  # certainty 0.70
    ]
    generator = _CountingGenerator()
    pipeline = RAGPipeline(_FakeRetriever(chunks), generator,
                           weak_threshold=0.55, min_chunk_score=0.52)
    result = pipeline.answer("anything")
    assert [c.chunk_id for c in generator.last_chunks] == ["c_good_1", "c_good_2"]
    assert [(s.marker, s.chunk_id) for s in result.sources] == [(1, "c_good_1"), (2, "c_good_2")]
    assert len(result.chunks) == 2  # the answer object keeps only what was used


def test_hybrid_fused_score_cannot_bypass_the_gate():
    # The exact trap from the Day 4 discussion: fused score 0.98 looks
    # confident, but the true cosine distance says "barely similar".
    trap = _chunk("c_bm25_only", 1, distance=0.95, score=0.98)
    generator = _CountingGenerator()
    pipeline = RAGPipeline(_FakeRetriever([trap]), generator,
                           weak_threshold=0.55, min_chunk_score=0.52)
    result = pipeline.answer("anything")
    assert result.weak_context is True and generator.calls == 0


def test_empty_retrieval_is_refused_gracefully():
    generator = _CountingGenerator()
    pipeline = RAGPipeline(_FakeRetriever([]), generator,
                           weak_threshold=0.55, min_chunk_score=0.52)
    result = pipeline.answer("anything")
    assert result.weak_context is True and generator.calls == 0
    assert result.retrieval_confidence == 0.0


def test_explicit_thresholds_override_defaults():
    generator = _CountingGenerator()
    strict = RAGPipeline(_FakeRetriever([_chunk("c1", 1, distance=0.4)]),  # certainty 0.8
                         generator, weak_threshold=0.9, min_chunk_score=0.52)
    assert strict.answer("anything").weak_context is True


if __name__ == "__main__":
    import sys

    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    sys.exit(1 if failures else 0)
