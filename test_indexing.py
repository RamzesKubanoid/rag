"""Self-checks for the shared indexing flow, including the
vector-index health probe (no Weaviate needed; a fake store records calls).

Run directly:      python test_indexing.py
Or with pytest:    pytest test_indexing.py -q
"""

from embedder import HashingEmbedder
from indexing import ensure_indexed


class _FakeStore:
    def __init__(self, count: int, search_hits: list):
        self._count = count
        self._search_hits = search_hits
        self.imported = None
        self.recreated = False

    def ensure_collection(self, model_label: str, recreate: bool = False):
        self.recreated = recreate

    def count(self) -> int:
        return self._count

    def search_by_vector(self, vector, top_k: int):
        return self._search_hits

    def index_chunks(self, chunks, vectors) -> int:
        assert len(chunks) == len(vectors)
        self.imported = len(chunks)
        self._count = len(chunks)
        return len(chunks)


def test_skips_import_when_populated_and_healthy():
    store = _FakeStore(count=24, search_hits=[{"properties": {}}])
    result = ensure_indexed(store, HashingEmbedder())
    assert result == 24 and store.imported is None


def test_indexes_when_collection_is_empty():
    store = _FakeStore(count=0, search_hits=[])
    result = ensure_indexed(store, HashingEmbedder())
    assert store.imported == result and result > 0  # real KB: 24 chunks


def test_heals_unhealthy_vector_index_by_recreating():
    # objects exist, but vector search returns nothing (unclean shutdown)
    store = _FakeStore(count=24, search_hits=[])
    result = ensure_indexed(store, HashingEmbedder())
    assert store.recreated is True, "heal must drop+recreate (re-import alone is not enough)"
    assert store.imported is not None and result == store.imported


def test_rebuild_recreates_and_imports():
    store = _FakeStore(count=24, search_hits=[{"properties": {}}])
    ensure_indexed(store, HashingEmbedder(), rebuild=True)
    assert store.recreated is True and store.imported is not None


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
