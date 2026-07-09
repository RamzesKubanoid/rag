"""Pydantic data models shared across the RAG pipeline.

Day 1 introduced `Document`, Day 2 adds `Chunk`. Later steps will add
models for retrieval results and the final structured answer.
"""

from pydantic import BaseModel, Field, computed_field


class TextItem(BaseModel):
    """Base class for any piece of text that needs size stats & previews."""

    text: str = Field(description="Cleaned text content")

    @computed_field
    @property
    def num_chars(self) -> int:
        """Length of the text in characters."""
        return len(self.text)

    @computed_field
    @property
    def num_words(self) -> int:
        """Number of whitespace-separated words in the text."""
        return len(self.text.split())

    def preview(self, max_chars: int = 100) -> str:
        """One-line preview of the text — handy in logs and debugging."""
        flat = " ".join(self.text.split())
        return flat[:max_chars] + ("..." if len(flat) > max_chars else "")


class Document(TextItem):
    """A single cleaned document loaded from the knowledge base (Day 1)."""

    doc_id: str = Field(description="Stable identifier, e.g. 'doc_001'")
    source: str = Field(description="File name the document was loaded from")
    path: str = Field(description="Path to the original file on disk")
    title: str = Field(description="Human-readable title (first line of the file)")


class Chunk(TextItem):
    """A retrieval-sized piece of a document, carrying citation metadata (Day 2).

    The text of every chunk except the first one starts with an overlap
    tail — the last sentences of the previous chunk — so that a thought
    cut at a chunk boundary survives intact in at least one chunk.
    """

    chunk_id: str = Field(description="Globally unique id, e.g. 'doc_001_chunk_000'")
    document_id: str = Field(description="doc_id of the parent document")
    source_name: str = Field(description="File name of the parent document")
    title: str = Field(description="Title of the parent document (useful for citations)")
    chunk_index: int = Field(ge=0, description="0-based position of the chunk inside its document")


class RetrievedChunk(Chunk):
    """A chunk returned by retrieval, enriched with relevance info (Day 4).

    Score semantics depend on the search mode:
      * vector search — score is Weaviate's certainty for cosine distance
        (0..1, higher = more similar); `distance` is the raw cosine
        distance (lower = closer).
      * hybrid search — score is the fused BM25+vector score: a ranking
        value normalized PER QUERY (relative score fusion), so it is not
        comparable to certainty and must not be used as an absolute
        confidence threshold; `distance` is the cosine distance computed
        client-side from the stored vector.
    """

    rank: int = Field(ge=1, description="1-based position in the result list")
    score: float = Field(description="Relevance score, higher is better")
    distance: float | None = Field(
        default=None, description="Cosine distance (lower is closer; client-side in hybrid mode)"
    )


class SourceRef(BaseModel):
    """One entry of the answer's source list (Day 5)."""

    marker: int = Field(ge=1, description="Citation number used in the answer, e.g. 1 for [1]")
    chunk_id: str
    source_name: str
    title: str
    score: float = Field(description="Retrieval score of this chunk (see RetrievedChunk)")


class RAGAnswer(BaseModel):
    """The final product of the pipeline: a grounded answer with sources (Day 5)."""

    question: str
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="The retrieved context the answer was built from"
    )
    model: str = Field(description="Name of the generation backend/model")
    used_retrieval: bool = True
    weak_context: bool = Field(
        default=False,
        description="True when retrieval confidence was too low and the pipeline "
        "refused without calling the LLM (Day 6)",
    )
    retrieval_confidence: float | None = Field(
        default=None,
        description="Best certainty among retrieved chunks (1 - cosine_distance/2); "
        "None for no-retrieval baseline answers",
    )


class AskRequest(BaseModel):
    """Request body for the API's /ask and /search endpoints (Day 7)."""

    question: str = Field(min_length=1, max_length=2000, description="The user question")
    top_k: int | None = Field(default=None, ge=1, le=20, description="How many chunks to retrieve")
    hybrid: bool = Field(default=False, description="Use BM25+vector hybrid retrieval")
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid mix: 0=keywords, 1=vectors")
