"""Self-checks for the Day 4 retriever's pure logic (no Weaviate needed —
the live path is exercised by `python main.py`, which runs a golden-set
retrieval demo against the real database).

Run directly:      python test_retriever.py
Or with pytest:    pytest test_retriever.py -q
"""

from retriever import _to_retrieved_chunks, normalize_question


def _hit(chunk_id: str, **extra) -> dict:
    return {
        "properties": {
            "chunk_id": chunk_id,
            "document_id": "doc_900",
            "source_name": "synthetic.txt",
            "title": "Synthetic",
            "chunk_index": 0,
            "text": "some chunk text",
        },
        **extra,
    }


def test_normalize_question_collapses_whitespace():
    assert normalize_question("  why   chunk\n\tdocuments? ") == "why chunk documents?"


def test_normalize_question_rejects_empty():
    for bad in ("", "   ", "\n\t"):
        try:
            normalize_question(bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_vector_mapping_uses_certainty_and_keeps_order():
    hits = [
        _hit("doc_900_chunk_000", distance=0.20, certainty=0.90),
        _hit("doc_900_chunk_001", distance=0.40, certainty=0.80),
    ]
    results = _to_retrieved_chunks(hits, mode="vector")
    assert [r.rank for r in results] == [1, 2]
    assert [r.chunk_id for r in results] == ["doc_900_chunk_000", "doc_900_chunk_001"]
    assert results[0].score == 0.90 and results[0].distance == 0.20


def test_vector_mapping_falls_back_to_distance():
    # certainty missing -> derived from cosine distance: 1 - d/2
    results = _to_retrieved_chunks([_hit("c", distance=0.30, certainty=None)], mode="vector")
    assert abs(results[0].score - 0.85) < 1e-9


def test_hybrid_mapping_uses_fused_score():
    results = _to_retrieved_chunks(
        [_hit("c", score=0.75, explain_score="bm25+vector")], mode="hybrid"
    )
    assert results[0].score == 0.75
    assert results[0].distance is None  # no stored vector -> no distance


def test_hybrid_mapping_passes_client_side_distance_through():
    results = _to_retrieved_chunks(
        [_hit("c", score=0.75, explain_score="x", distance=0.42)], mode="hybrid"
    )
    assert results[0].score == 0.75 and results[0].distance == 0.42


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
