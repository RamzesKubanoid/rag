"""API contract tests with fake dependencies (no Weaviate / LLM).

Run directly:      python test_api.py
Or with pytest:    pytest test_api.py -q
"""

import os

os.environ["RAG_API_SKIP_INIT"] = "1"  # must be set before importing api

from fastapi.testclient import TestClient

from api import app, get_pipeline, get_retriever, get_store
from prompts import NO_ANSWER_SENTENCE
from schemas import RAGAnswer, RetrievedChunk, SourceRef

_CHUNK = RetrievedChunk(
    chunk_id="doc_005_chunk_000",
    document_id="doc_005",
    source_name="05_chunking_strategies.txt",
    title="Chunking Strategies for RAG",
    chunk_index=0,
    text="Chunking splits documents into smaller pieces.",
    rank=1,
    score=0.91,
    distance=0.18,
)


class _FakeStore:
    collection_name = "KnowledgeChunk"

    def count(self):
        return 24


class _FakeRetriever:
    def retrieve(self, question, top_k=None, hybrid=False, alpha=0.5):
        if not question.strip():
            raise ValueError("Question is empty")
        return [_CHUNK]


class _FakePipeline:
    weak_threshold = 0.55
    min_chunk_score = 0.52

    class generator:  # noqa: N801 - mimics the attribute shape
        model_name = "fake-llm"

    def answer(self, question, top_k=None, hybrid=False, alpha=0.5):
        if not question.strip():
            raise ValueError("Question is empty")
        if "france" in question.lower():
            return RAGAnswer(
                question=question,
                answer=f"{NO_ANSWER_SENTENCE} (below threshold)",
                sources=[], chunks=[_CHUNK], model="fake-llm",
                weak_context=True, retrieval_confidence=0.53,
            )
        return RAGAnswer(
            question=question,
            answer="Chunking makes retrieval precise [1].",
            sources=[SourceRef(marker=1, chunk_id=_CHUNK.chunk_id,
                               source_name=_CHUNK.source_name,
                               title=_CHUNK.title, score=_CHUNK.score)],
            chunks=[_CHUNK], model="fake-llm",
            weak_context=False, retrieval_confidence=0.91,
        )


app.dependency_overrides[get_store] = lambda: _FakeStore()
app.dependency_overrides[get_retriever] = lambda: _FakeRetriever()
app.dependency_overrides[get_pipeline] = lambda: _FakePipeline()

client = TestClient(app)


def test_health_reports_index_and_backends():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok" and payload["objects"] == 24
    assert payload["collection"] == "KnowledgeChunk" and payload["llm"] == "fake-llm"


def test_ask_returns_grounded_answer_with_sources():
    response = client.post("/ask", json={"question": "Why chunk documents?"})
    assert response.status_code == 200
    payload = response.json()
    assert "[1]" in payload["answer"]
    assert payload["sources"][0]["marker"] == 1
    assert payload["sources"][0]["source_name"] == "05_chunking_strategies.txt"
    assert payload["weak_context"] is False
    assert abs(payload["retrieval_confidence"] - 0.91) < 1e-9


def test_ask_weak_context_is_http_200_with_flag():
    response = client.post("/ask", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["weak_context"] is True and payload["sources"] == []
    assert NO_ANSWER_SENTENCE in payload["answer"]


def test_search_returns_scored_chunks():
    response = client.post("/search", json={"question": "chunking", "top_k": 3})
    assert response.status_code == 200
    hits = response.json()
    assert hits[0]["chunk_id"] == "doc_005_chunk_000" and hits[0]["score"] == 0.91


def test_validation_and_error_paths():
    assert client.post("/ask", json={"question": ""}).status_code == 422       # pydantic min_length
    assert client.post("/ask", json={"question": "   "}).status_code == 400    # normalize_question
    assert client.post("/ask", json={"question": "x", "alpha": 3}).status_code == 422


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
