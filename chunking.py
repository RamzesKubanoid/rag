"""Day 2 — splitting documents into retrieval-sized chunks.

Strategy: paragraph-aware packing with a sentence-level fallback.

  1. Split the cleaned document into paragraphs (blank-line separated —
     Day 1 deliberately preserved these boundaries).
  2. Any paragraph longer than `max_chars` is split into sentences; any
     monster sentence without punctuation is hard-split by words.
  3. Units are greedily packed into chunks whose NEW content is at most
     `max_chars` long, so a chunk never breaks inside a paragraph unless
     the paragraph itself was oversized.
  4. Every chunk after the first is prefixed with the last sentences of
     the previous chunk, up to `overlap_chars`, so an idea cut at a
     boundary still appears intact in at least one chunk.

Size contract:
  - new content per chunk       <= max_chars
  - full chunk text             <= max_chars + overlap_chars + 2
    (the +2 is the separator between the overlap tail and new content)

Why these defaults: chunks of ~1000 characters (~150-200 English words)
are small enough for precise similarity matching but big enough to keep
a self-contained thought; the ~20% overlap protects boundary sentences.
Both values are parameters and should be tuned on real queries later.
"""

import logging
import re

from schemas import Chunk, Document

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 1000       # max NEW content per chunk (~150-200 words)
DEFAULT_OVERLAP_CHARS = 200    # ~20% of max_chars, carried over from the previous chunk
DEFAULT_MIN_CHUNK_CHARS = 200  # trailing chunks smaller than this are merged back when they fit

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------
# Low-level splitting helpers
# ----------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split cleaned text on blank lines (paragraph boundaries)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for prose; abbreviations
    like 'e.g.' may fool it, which is acceptable for this project."""
    flat = " ".join(text.split())
    return [s.strip() for s in _SENTENCE_END.split(flat) if s.strip()]


def _pack(units: list[str], max_chars: int, separator: str) -> list[list[str]]:
    """Greedily pack units into groups whose joined length is <= max_chars.

    A single unit longer than max_chars still gets its own group; callers
    are expected to pre-split such units.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    size = 0
    for unit in units:
        extra = len(unit) + (len(separator) if current else 0)
        if current and size + extra > max_chars:
            groups.append(current)
            current, size = [unit], len(unit)
        else:
            current.append(unit)
            size += extra
    if current:
        groups.append(current)
    return groups


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort split of a huge sentence by words (or raw slicing)."""
    units: list[str] = []
    for word in text.split():
        if len(word) > max_chars:  # pathological single token
            units.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
        else:
            units.append(word)
    return [" ".join(group) for group in _pack(units, max_chars, " ")]


def _split_oversized_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """Split a too-long paragraph into sentence groups of <= max_chars."""
    units: list[str] = []
    for sentence in _split_sentences(paragraph):
        if len(sentence) > max_chars:
            units.extend(_hard_split(sentence, max_chars))
        else:
            units.append(sentence)
    return [" ".join(group) for group in _pack(units, max_chars, " ")]


def _document_units(document: Document, max_chars: int) -> list[str]:
    """Turn a document into packing units, each guaranteed <= max_chars."""
    units: list[str] = []
    for paragraph in _split_paragraphs(document.text):
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            logger.debug(
                "Oversized paragraph (%d chars) in %s — splitting by sentences",
                len(paragraph), document.source,
            )
            units.extend(_split_oversized_paragraph(paragraph, max_chars))
    return units


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Return the last sentences of `text`, at most `overlap_chars` long."""
    if overlap_chars <= 0:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= overlap_chars:
        return flat
    tail: list[str] = []
    size = 0
    for sentence in reversed(_split_sentences(flat)):
        extra = len(sentence) + (1 if tail else 0)
        if size + extra > overlap_chars:
            break
        tail.append(sentence)
        size += extra
    if tail:
        return " ".join(reversed(tail))
    # even the last single sentence is over budget: cut at a word boundary
    cut = flat[-overlap_chars:]
    first_space = cut.find(" ")
    return cut[first_space + 1:] if first_space != -1 else cut


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def split_document(
    document: Document,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split one document into overlapping chunks with citation metadata."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be >= 0 and smaller than max_chars")

    units = _document_units(document, max_chars)
    groups = _pack(units, max_chars, "\n\n")

    # A tiny trailing group is merged into the previous one when it fits,
    # so retrieval does not have to deal with near-empty chunks.
    if len(groups) > 1:
        last_len = len("\n\n".join(groups[-1]))
        prev_len = len("\n\n".join(groups[-2]))
        if last_len < min_chunk_chars and prev_len + 2 + last_len <= max_chars:
            groups[-2].extend(groups.pop())

    chunks: list[Chunk] = []
    previous_core = ""
    for index, group in enumerate(groups):
        core = "\n\n".join(group)
        if index > 0 and overlap_chars > 0:
            tail = _overlap_tail(previous_core, overlap_chars)
            text = f"{tail}\n\n{core}" if tail else core
        else:
            text = core
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}_chunk_{index:03d}",
                document_id=document.doc_id,
                source_name=document.source,
                title=document.title,
                chunk_index=index,
                text=text,
            )
        )
        previous_core = core  # overlap always comes from NEW content only

    logger.debug(
        "%s: %d chars -> %d chunks", document.source, document.num_chars, len(chunks)
    )
    return chunks


def split_documents(
    documents: list[Document],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[Chunk]:
    """Split every document and return one flat list of chunks."""
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(
            split_document(document, max_chars, overlap_chars, min_chunk_chars)
        )
    logger.info(
        "Chunking: %d documents -> %d chunks (max_chars=%d, overlap_chars=%d)",
        len(documents), len(all_chunks), max_chars, overlap_chars,
    )
    return all_chunks
