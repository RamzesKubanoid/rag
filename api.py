"""FastAPI service exposing the RAG pipeline over HTTP.

Run (after configuring .env and starting Weaviate):
    uvicorn api:app --reload
    # or simply: python api.py

Endpoints:
    GET  /health   service status + index/backends info
    POST /search   retrieval only — inspect the context for a question
    POST /ask      the full RAG cycle — grounded answer with sources

Interactive documentation: http://localhost:8000/docs

On startup the service connects to Weaviate and indexes the knowledge
base automatically if the collection is empty; otherwise the existing
index is reused (rebuild via the CLI: `python main.py --rebuild`).
"""

import logging
import os
from contextlib import asynccontextmanager

try:  # .env support is optional but convenient
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from fastapi import Depends, FastAPI, HTTPException, Request

from embedder import get_embedder
from generator import get_generator
from indexing import ensure_indexed
from rag_pipeline import RAGPipeline
from retriever import Retriever
from schemas import AskRequest, RAGAnswer, RetrievedChunk
from weaviate_store import WeaviateStore
from fastapi.responses import RedirectResponse

logger = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline once at startup; close Weaviate at shutdown."""
    if os.getenv("RAG_API_SKIP_INIT") == "1":  # test hook: dependencies are overridden
        yield
        return

    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    store = WeaviateStore.connect()
    embedder = get_embedder()
    count = ensure_indexed(store, embedder)

    app.state.store = store
    app.state.embedder_name = embedder.model_name
    app.state.retriever = Retriever(store, embedder)
    app.state.pipeline = RAGPipeline(app.state.retriever, get_generator())
    logger.info("API ready — %d chunks indexed, docs at /docs", count)

    yield
    store.close()


app = FastAPI(
    title="RAG on Weaviate — Knowledge Base Q&A",
    description="RAG mini-product: retrieval from Weaviate, "
    "grounded LLM answers with sources, weak-context protection.",
    version="1.0.0",
    lifespan=lifespan,
)


# -- dependencies (overridable in tests) --------------------------------

def get_store(request: Request):
    return request.app.state.store


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


# -- endpoints -----------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Convenience: opening the service root in a browser lands on the docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health(store=Depends(get_store), pipeline=Depends(get_pipeline)) -> dict:
    """Service status, index size and active backends."""
    return {
        "status": "ok",
        "collection": store.collection_name,
        "objects": store.count(),
        "embedder": app.state.embedder_name if hasattr(app.state, "embedder_name") else None,
        "llm": pipeline.generator.model_name,
        "weak_threshold": pipeline.weak_threshold,
        "min_chunk_score": pipeline.min_chunk_score,
    }


@app.post("/search", response_model=list[RetrievedChunk])
def search(body: AskRequest, retriever: Retriever = Depends(get_retriever)):
    """Retrieval only: the top-k chunks with scores — no LLM call."""
    try:
        return retriever.retrieve(
            body.question, top_k=body.top_k, hybrid=body.hybrid, alpha=body.alpha
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=RAGAnswer)
def ask(body: AskRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    """The full RAG cycle: grounded answer + [n]-cited sources.

    A weak-context refusal is a NORMAL response (HTTP 200) with
    `weak_context: true` and an empty source list — not an error.
    """
    try:
        return pipeline.answer(
            body.question, top_k=body.top_k, hybrid=body.hybrid, alpha=body.alpha
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=int(os.getenv("API_PORT", "8000")))
