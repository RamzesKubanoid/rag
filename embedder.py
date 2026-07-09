"""Embeddings for chunks and queries.

Two interchangeable backends behind one interface:

  * OpenAIEmbedder  — real semantic embeddings via the OpenAI API (or any
    OpenAI-compatible endpoint through OPENAI_BASE_URL). Used when
    OPENAI_API_KEY is set.
  * HashingEmbedder — offline, deterministic stand-in (hashed bag of
    words, L2-normalized). Needs no network or key, so the pipeline can
    be developed and tested for free. It captures keyword overlap only,
    NOT real semantics — switch to OpenAI for meaningful retrieval.

Both documents and queries MUST be embedded with the same backend and
model; vectors from different models live in incompatible spaces.
"""

import hashlib
import logging
import math
import os
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# Small English stopword list shared by the offline components: without it,
# function words ("what is the of ...") dominate lexical similarity.
STOPWORDS = frozenset(
    """a about after all also an and any are as at be been being but by can
    could did do does for from had has have how i if in into is it its just
    may might more most not of on or our so some such than that the then
    there these they this to too very was we what when where which who why
    will with would you your""".split()
)

# Known dimensions so we don't need a probe request for common models.
_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class Embedder(ABC):
    """Common interface: turn texts into fixed-size float vectors."""

    model_name: str
    dimension: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, preserving order."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]


class OpenAIEmbedder(Embedder):
    """Embeddings via the OpenAI API or any compatible endpoint."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
    ) -> None:
        from openai import OpenAI  # local import keeps the dependency optional

        self.model_name = model
        self.batch_size = batch_size
        # Per-component overrides first, shared OPENAI_* variables as fallback —
        # lets embeddings and the LLM use different providers (e.g. Gemini
        # embeddings + a Hugging Face-served LLM).
        self._client = OpenAI(
            api_key=api_key or os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None,
        )
        self.dimension = _OPENAI_DIMENSIONS.get(model, 0)  # 0 = learn from first call

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self._client.embeddings.create(model=self.model_name, input=batch)
            data = list(response.data)
            if len(data) != len(batch):
                raise RuntimeError(
                    f"Embedding API returned {len(data)} vectors for {len(batch)} inputs"
                )
            # The OpenAI spec returns items in input order AND with an integer
            # `index` field. Some compatible providers (e.g. Gemini) leave the
            # index null, so sort only when every index is really an int and
            # trust the returned order otherwise.
            if all(isinstance(item.index, int) for item in data):
                data.sort(key=lambda item: item.index)
            vectors.extend(item.embedding for item in data)
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors


class HashingEmbedder(Embedder):
    """Offline stand-in: signed hashed bag-of-words, L2-normalized.

    Deterministic (sha1-based, not Python's randomized hash()), so the
    same text always produces the same vector across runs and machines.
    Cosine similarity of these vectors reflects word overlap — enough to
    exercise the whole pipeline, not enough for semantic search quality.
    """

    _TOKEN = re.compile(r"[a-zA-Z0-9]+")

    _STOPWORDS = STOPWORDS

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.model_name = f"hashing-bow-{dimension}d-v2"

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in self._TOKEN.findall(text.lower()):
            if len(token) < 2 or token in self._STOPWORDS:
                continue
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0  # signed hashing trick
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:  # empty / non-alphanumeric text: return a safe unit vector
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]


def get_embedder() -> Embedder:
    """Pick the embedding backend from environment configuration.

    EMBEDDING_BACKEND = auto (default) | openai | hash
      auto   -> OpenAI when OPENAI_API_KEY is set, otherwise hashing.
      openai -> force OpenAI (fails later without a key).
      hash   -> force the offline embedder.
    EMBEDDING_MODEL sets the OpenAI model (default text-embedding-3-small).
    """
    backend = os.getenv("EMBEDDING_BACKEND", "auto").lower()
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_OPENAI_MODEL)

    if backend not in {"auto", "openai", "hash"}:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r} (use auto|openai|hash)")

    has_key = bool(os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if backend == "openai" or (backend == "auto" and has_key):
        logger.info("Embedding backend: OpenAI-compatible API (%s)", model)
        return OpenAIEmbedder(model=model)

    if backend == "auto":
        logger.warning(
            "No EMBEDDING_API_KEY / OPENAI_API_KEY set — falling back to the offline HashingEmbedder. "
            "Retrieval will match keywords only; set the key (see .env.example) "
            "for real semantic embeddings."
        )
    else:
        logger.info("Embedding backend: offline HashingEmbedder")
    return HashingEmbedder()
