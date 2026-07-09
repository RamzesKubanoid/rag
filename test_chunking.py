"""Self-checks for the Day 2 chunker.

Run directly:      python test_chunking.py
Or with pytest:    pytest test_chunking.py -q
"""

from chunking import split_document, split_documents
from schemas import Document


# ----------------------------------------------------------------------
# Helpers to build synthetic documents
# ----------------------------------------------------------------------

def _make_document(text: str, doc_id: str = "doc_900") -> Document:
    return Document(
        doc_id=doc_id,
        source="synthetic.txt",
        path="synthetic.txt",
        title="Synthetic",
        text=text,
    )


def _long_document(paragraphs: int = 40, sentences_per_paragraph: int = 4) -> Document:
    """A long document with unique markers so we can track every paragraph."""
    parts = []
    for p in range(paragraphs):
        sentences = [
            f"Paragraph PARA-{p:02d} sentence {s} talks about topic number "
            f"{p * 10 + s} in some level of detail."
            for s in range(sentences_per_paragraph)
        ]
        parts.append(" ".join(sentences))
    return _make_document("\n\n".join(parts))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def test_short_document_stays_single_chunk():
    doc = _make_document("A tiny document.\n\nJust two paragraphs.")
    chunks = split_document(doc, max_chars=1000, overlap_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text == doc.text
    assert chunks[0].chunk_id == "doc_900_chunk_000"
    assert chunks[0].document_id == "doc_900"
    assert chunks[0].source_name == "synthetic.txt"
    assert chunks[0].chunk_index == 0


def test_long_document_is_split_within_limits():
    doc = _long_document()
    max_chars, overlap = 800, 150
    chunks = split_document(doc, max_chars=max_chars, overlap_chars=overlap)
    assert len(chunks) > 5, "a long document must produce many chunks"
    for chunk in chunks:
        # contract: new content <= max_chars, full text <= max + overlap + separator
        assert chunk.num_chars <= max_chars + overlap + 2, chunk.chunk_id


def test_consecutive_chunks_overlap():
    doc = _long_document()
    chunks = split_document(doc, max_chars=800, overlap_chars=150)
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_flat = " ".join(prev.text.split())
        nxt_flat = " ".join(nxt.text.split())
        probe = nxt_flat[:40]  # the start of a chunk must repeat the previous one
        assert probe in prev_flat, f"no overlap between {prev.chunk_id} and {nxt.chunk_id}"


def test_no_paragraph_lost():
    total = 25
    doc = _long_document(paragraphs=total)
    chunks = split_document(doc, max_chars=700, overlap_chars=100)
    combined = " ".join(" ".join(c.text.split()) for c in chunks)
    for p in range(total):
        assert f"PARA-{p:02d}" in combined, f"paragraph {p} disappeared during chunking"


def test_oversized_sentence_is_hard_split():
    monster = ("word " * 1200).strip()  # ~6000 chars without any sentence punctuation
    doc = _make_document(monster)
    chunks = split_document(doc, max_chars=500, overlap_chars=80)
    assert len(chunks) >= 10
    assert all(c.num_chars <= 500 + 80 + 2 for c in chunks)


def test_chunk_ids_are_sequential_and_unique():
    docs = [
        _long_document(paragraphs=10),
        _make_document("Short one.", doc_id="doc_901"),
    ]
    chunks = split_documents(docs, max_chars=600, overlap_chars=100)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be globally unique"
    for doc in docs:
        indices = [c.chunk_index for c in chunks if c.document_id == doc.doc_id]
        assert indices == list(range(len(indices))), "chunk_index must be sequential"


def test_invalid_overlap_raises():
    doc = _make_document("Some text.")
    try:
        split_document(doc, max_chars=100, overlap_chars=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when overlap_chars >= max_chars")


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
