"""Storing chunks and their vectors in Weaviate.

Design decisions:
  * "Bring your own vectors": the collection has no vectorizer module;
    embeddings are computed by embedder.py and sent explicitly. Every
    step of the pipeline stays visible and swappable.
  * Deterministic object UUIDs derived from chunk_id (uuid5), so
    re-indexing the same knowledge base UPSERTS objects instead of
    creating duplicates. Note: if chunking parameters change, chunk_ids
    change too and stale objects from the old run would remain — that is
    what ensure_collection(recreate=True) / `python main.py --rebuild`
    is for.
  * Cosine distance for the HNSW index — the natural pairing for
    normalized text embeddings.

Connection modes (WEAVIATE_MODE):
  * local    — a Weaviate server started with `docker compose up -d`
               (see docker-compose.yml); default.
  * embedded — Weaviate Embedded: the client downloads a server binary
               on first use and runs it as a subprocess. No Docker
               needed; handy for CI and quick experiments.
"""

import logging
import math
import os

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.util import generate_uuid5

from schemas import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "KnowledgeChunk"


def _cosine_distance(a: list[float], b: list[float]) -> float | None:
    """Cosine distance between two vectors (1 - cosine similarity)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return 1.0 - dot / (norm_a * norm_b)


def _vector_config_kwargs() -> dict:
    """Collection vector settings for both new and older v4 clients."""
    hnsw_cosine = Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
    if hasattr(Configure, "Vectors"):  # weaviate-client >= 4.16
        return {"vector_config": Configure.Vectors.self_provided(vector_index_config=hnsw_cosine)}
    return {  # older 4.x clients
        "vectorizer_config": Configure.Vectorizer.none(),
        "vector_index_config": hnsw_cosine,
    }


class WeaviateStore:
    """Thin wrapper around one Weaviate collection holding knowledge chunks."""

    def __init__(self, client: weaviate.WeaviateClient, collection_name: str = COLLECTION_NAME):
        self._client = client
        self.collection_name = collection_name

    # -- connection ----------------------------------------------------

    @classmethod
    def connect(cls) -> "WeaviateStore":
        """Connect according to WEAVIATE_MODE (local docker | embedded)."""
        mode = os.getenv("WEAVIATE_MODE", "local").lower()
        if mode == "embedded":
            logger.info("Connecting to embedded Weaviate (no Docker)")
            client = weaviate.connect_to_embedded(
                persistence_data_path=os.getenv("WEAVIATE_DATA_PATH", "./weaviate_data"),
                environment_variables={
                    "LOG_LEVEL": os.getenv("WEAVIATE_LOG_LEVEL", "error"),
                    # Single-node embedded server: advertise on loopback so it
                    # also starts in containers/CI without a private IP.
                    "CLUSTER_ADVERTISE_ADDR": "127.0.0.1",
                    "DISABLE_TELEMETRY": "true",
                },
            )
        elif mode == "local":
            host = os.getenv("WEAVIATE_HOST", "localhost")
            port = int(os.getenv("WEAVIATE_PORT", "8080"))
            grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
            logger.info("Connecting to Weaviate at %s:%d (gRPC %d)", host, port, grpc_port)
            client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
        else:
            raise ValueError(f"Unknown WEAVIATE_MODE: {mode!r} (use local|embedded)")
        logger.info("Weaviate is ready (server %s)", client.get_meta().get("version", "?"))
        return cls(client)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WeaviateStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- schema ---------------------------------------------------------

    def ensure_collection(self, model_label: str, recreate: bool = False) -> None:
        """Create the chunk collection if needed; optionally drop it first.

        The embedding model name is written into the collection
        description so a later run with a different embedder can be
        detected — mixed vectors would silently ruin retrieval.
        """
        if recreate and self._client.collections.exists(self.collection_name):
            logger.info("Dropping collection '%s' (rebuild requested)", self.collection_name)
            self._client.collections.delete(self.collection_name)

        if self._client.collections.exists(self.collection_name):
            try:
                description = (
                    self._client.collections.get(self.collection_name).config.get().description or ""
                )
                if model_label not in description:
                    logger.warning(
                        "Collection '%s' was built with a different embedder (%s) than the "
                        "current one (%s). Vectors are incompatible — run with --rebuild.",
                        self.collection_name, description, model_label,
                    )
            except Exception:  # description check is best-effort only
                logger.debug("Could not read collection description", exc_info=True)
            return

        logger.info("Creating collection '%s'", self.collection_name)
        self._client.collections.create(
            name=self.collection_name,
            description=f"RAG knowledge chunks (embeddings: {model_label})",
            properties=[
                Property(name="chunk_id", data_type=DataType.TEXT),
                Property(name="document_id", data_type=DataType.TEXT),
                Property(name="source_name", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            **_vector_config_kwargs(),
        )

    def drop(self) -> None:
        """Delete the collection entirely (schema + data)."""
        if self._client.collections.exists(self.collection_name):
            self._client.collections.delete(self.collection_name)
            logger.info("Collection '%s' deleted", self.collection_name)

    # -- data -----------------------------------------------------------

    def index_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Batch-import chunks with their vectors. Idempotent by design:
        the object UUID is uuid5(chunk_id), so re-importing the same
        chunks overwrites objects instead of duplicating them."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        collection = self._client.collections.get(self.collection_name)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                batch.add_object(
                    uuid=generate_uuid5(chunk.chunk_id),
                    properties={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "source_name": chunk.source_name,
                        "title": chunk.title,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    },
                    vector=vector,
                )

        failed = collection.batch.failed_objects
        if failed:
            logger.error("%d objects failed to import; first error: %s",
                         len(failed), failed[0].message)
            raise RuntimeError("Weaviate batch import reported failures")

        logger.info("Imported %d chunks into '%s'", len(chunks), self.collection_name)
        return len(chunks)

    # -- verification ----------------------------------------------------

    def count(self) -> int:
        """Number of objects currently stored in the collection."""
        collection = self._client.collections.get(self.collection_name)
        result = collection.aggregate.over_all(total_count=True)
        return int(result.total_count or 0)

    def sample(self, limit: int = 2) -> list[dict]:
        """Fetch a few stored objects to prove data + vectors are there."""
        collection = self._client.collections.get(self.collection_name)
        response = collection.query.fetch_objects(limit=limit, include_vector=True)
        samples = []
        for obj in response.objects:
            vector = obj.vector
            if isinstance(vector, dict):  # newer clients: {"default": [...]}
                vector = next(iter(vector.values()), [])
            samples.append(
                {"uuid": str(obj.uuid), "vector_dim": len(vector or []), **obj.properties}
            )
        return samples


    # -- retrieval (Day 4) ------------------------------------------------

    def search_by_vector(self, vector: list[float], top_k: int) -> list[dict]:
        """Pure semantic search: top-k nearest neighbours of `vector`.

        Returns raw hits (properties + distance + certainty); mapping to
        pydantic models is the retriever's job.
        """
        from weaviate.classes.query import MetadataQuery

        collection = self._client.collections.get(self.collection_name)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, certainty=True),
        )
        return [
            {
                "properties": dict(obj.properties),
                "distance": obj.metadata.distance,
                "certainty": obj.metadata.certainty,
            }
            for obj in response.objects
        ]

    def search_hybrid(
        self, query_text: str, vector: list[float], top_k: int, alpha: float = 0.5
    ) -> list[dict]:
        """Hybrid search: BM25 keyword score blended with vector similarity.

        alpha=0 is pure keyword search, alpha=1 pure vector search.
        Because this collection brings its own vectors (no vectorizer
        module), the query vector must be supplied explicitly.

        The hybrid API returns only the fused score (a per-query,
        relatively normalized ranking value), so the geometric cosine
        distance of each hit is computed here client-side from the
        stored vector — useful as an absolute similarity signal.
        """
        from weaviate.classes.query import MetadataQuery

        collection = self._client.collections.get(self.collection_name)
        response = collection.query.hybrid(
            query=query_text,
            vector=vector,
            alpha=alpha,
            limit=top_k,
            include_vector=True,
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )
        hits: list[dict] = []
        for obj in response.objects:
            stored = obj.vector
            if isinstance(stored, dict):  # newer clients: {"default": [...]}
                stored = next(iter(stored.values()), None)
            hits.append(
                {
                    "properties": dict(obj.properties),
                    "score": obj.metadata.score,
                    "explain_score": obj.metadata.explain_score,
                    "distance": _cosine_distance(vector, stored) if stored else None,
                }
            )
        return hits
