"""Self-checks for Day 5: prompt building, the offline extractive
generator and the pipeline wiring. No Weaviate or LLM required — the
live path is exercised by `python main.py`.

Run directly:      python test_pipeline.py
Or with pytest:    pytest test_pipeline.py -q
"""

from generator import ExtractiveGenerator
from prompts import NO_ANSWER_SENTENCE, SYSTEM_PROMPT_RAG, build_rag_user_prompt, format_context
from rag_pipeline import RAGPipeline
from schemas import RetrievedChunk


def _chunk(chunk_id: str, text: str, rank: int = 1, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id.rsplit("_chunk_", 1)[0],
        source_name=f"{chunk_id}.txt",
        title="Test Doc",
        chunk_index=0,
        text=text,
        rank=rank,
        score=score,
    )


CHUNKS = [
    _chunk(
        "doc_005_chunk_000",
        "Chunking splits documents into smaller pieces before indexing. "
        "Small chunks give precise retrieval matches.",
        rank=1,
    ),
    _chunk(
        "doc_002_chunk_000",
        "An embedding is a numeric vector that captures meaning. "
        "Bananas are yellow fruit.",
        rank=2,
        score=0.6,
    ),
]


def test_format_context_numbers_and_labels_fragments():
    block = format_context(CHUNKS)
    assert "[1] (source: doc_005_chunk_000.txt, chunk: doc_005_chunk_000)" in block
    assert "[2] (source: doc_002_chunk_000.txt, chunk: doc_002_chunk_000)" in block
    assert block.index("[1]") < block.index("[2]")


def test_rag_prompt_contains_question_context_and_rules():
    prompt = build_rag_user_prompt("Why chunk documents?", CHUNKS)
    assert "Why chunk documents?" in prompt
    assert "Chunking splits documents" in prompt
    assert "ONLY the numbered context fragments" in SYSTEM_PROMPT_RAG
    assert NO_ANSWER_SENTENCE in SYSTEM_PROMPT_RAG  # explicit anti-fabrication rule


def test_extractive_answers_with_citations():
    answer = ExtractiveGenerator().answer_with_context("Why chunk documents into pieces?", CHUNKS)
    assert "[1]" in answer                      # cites the relevant chunk
    assert "Chunking splits documents" in answer
    assert "Bananas" not in answer              # irrelevant sentence not extracted


def test_extractive_declines_on_unrelated_context():
    answer = ExtractiveGenerator().answer_with_context("What is the capital of France?", CHUNKS)
    assert answer == NO_ANSWER_SENTENCE


def test_extractive_deduplicates_overlap_sentences():
    duplicated = [
        CHUNKS[0],
        _chunk("doc_005_chunk_001", "Small chunks give precise retrieval matches. "
                                    "Overlap protects boundary sentences.", rank=2),
    ]
    answer = ExtractiveGenerator().answer_with_context("Do small chunks give precise matches?", duplicated)
    assert answer.count("Small chunks give precise retrieval matches.") == 1


class _FakeRetriever:
    def retrieve(self, question, top_k=None, hybrid=False, alpha=0.5):
        return CHUNKS


def test_pipeline_assembles_answer_and_sources():
    pipeline = RAGPipeline(_FakeRetriever(), ExtractiveGenerator())
    result = pipeline.answer("Why chunk documents into pieces?")
    assert result.used_retrieval is True
    assert result.model == "extractive-fallback"
    assert [s.marker for s in result.sources] == [1, 2]          # markers mirror context order
    assert result.sources[0].chunk_id == "doc_005_chunk_000"
    assert len(result.chunks) == 2 and result.answer


def test_pipeline_baseline_has_no_sources():
    pipeline = RAGPipeline(_FakeRetriever(), ExtractiveGenerator())
    result = pipeline.answer_baseline("Why chunk documents?")
    assert result.used_retrieval is False
    assert result.sources == [] and result.chunks == []


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
