"""Self-checks for the Day 3 embedder (offline backend only — the OpenAI
backend needs an API key and is exercised on the real machine instead).

Run directly:      python test_embedder.py
Or with pytest:    pytest test_embedder.py -q
"""

import math

from embedder import HashingEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are already L2-normalized


def test_deterministic_across_calls():
    embedder = HashingEmbedder()
    first = embedder.embed_query("Weaviate stores vectors and metadata.")
    second = embedder.embed_query("Weaviate stores vectors and metadata.")
    assert first == second


def test_dimension_and_unit_norm():
    embedder = HashingEmbedder(dimension=384)
    vector = embedder.embed_query("Chunking splits documents into pieces.")
    assert len(vector) == 384
    norm = math.sqrt(sum(v * v for v in vector))
    assert abs(norm - 1.0) < 1e-9


def test_related_texts_are_more_similar_than_unrelated():
    embedder = HashingEmbedder()
    query = embedder.embed_query("How does a vector database search embeddings?")
    related = embedder.embed_query("A vector database stores embeddings and answers similarity search queries.")
    unrelated = embedder.embed_query("The recipe requires two eggs, flour, butter and a pinch of salt.")
    assert _cosine(query, related) > _cosine(query, unrelated)


def test_batch_preserves_order_and_handles_empty_text():
    embedder = HashingEmbedder()
    texts = ["first text about retrieval", "", "third text about retrieval"]
    vectors = embedder.embed_texts(texts)
    assert len(vectors) == 3
    assert vectors[0] == embedder.embed_query(texts[0])          # order preserved
    assert abs(sum(v * v for v in vectors[1]) - 1.0) < 1e-9      # empty text -> safe unit vector
    assert _cosine(vectors[0], vectors[2]) > 0.5                 # near-identical texts agree


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
