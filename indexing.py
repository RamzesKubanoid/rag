"""The shared indexing flow used by both the CLI and the API.

One function owns the "make sure the knowledge base is in Weaviate"
logic: create the collection if needed, then load -> chunk -> embed ->
import, but only when the collection is empty or a refresh is forced.
"""

import logging
import time

from chunking import split_documents
from embedder import Embedder
from loader import load_knowledge_base
from weaviate_store import WeaviateStore

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = "knowledge_base"


def ensure_indexed(
    store: WeaviateStore,
    embedder: Embedder,
    kb_dir: str = KNOWLEDGE_BASE_DIR,
    reindex: bool = False,
    rebuild: bool = False,
) -> int:
    """Guarantee an up-to-date index; return the object count afterwards.

    * rebuild=True  — drop the collection first (chunking/embedder changed)
    * reindex=True  — re-import into the existing collection (KB edited)
    * neither       — index only if the collection is empty
    """
    store.ensure_collection(model_label=embedder.model_name, recreate=rebuild)

    existing = store.count()
    if existing and not (rebuild or reindex):
        # Cheap health probe (one embedding call): a populated collection
        # whose vector index returns nothing — possible after an unclean
        # shutdown right after import — would silently turn every question
        # into a "weak context" refusal. Detect it and heal by re-importing.
        probe = store.search_by_vector(embedder.embed_query("index health probe"), top_k=1)
        if probe:
            logger.info(
                "Index already populated (%d objects) — skipping import "
                "(--reindex after editing the KB, --rebuild after changing "
                "chunking or the embedding model)", existing,
            )
            return existing
        logger.warning(
            "Collection holds %d objects but vector search returns nothing — "
            "the vector index looks unhealthy (unclean shutdown?). "
            "Recreating the collection to heal it (re-importing alone does "
            "not repair a corrupted HNSW index).", existing,
        )
        store.ensure_collection(model_label=embedder.model_name, recreate=True)

    logger.info("Indexing the knowledge base from '%s/'", kb_dir)
    documents = load_knowledge_base(kb_dir)
    chunks = split_documents(documents)

    started = time.perf_counter()
    vectors = embedder.embed_texts([chunk.text for chunk in chunks])
    logger.info(
        "Embedded %d chunks with %s in %.2fs (dimension=%d)",
        len(vectors), embedder.model_name, time.perf_counter() - started, embedder.dimension,
    )
    store.index_chunks(chunks, vectors)
    return store.count()
