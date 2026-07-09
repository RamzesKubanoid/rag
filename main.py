"""RAG on Weaviate — application entry point.

Day 6 scope: the full RAG cycle protected by weak-context guards — a
per-chunk noise filter and a confidence gate that refuses BEFORE the LLM
when retrieval is poor — plus an evaluation harness over answerable and
unanswerable question sets (`--evaluate`).

Usage (from the project root):
    python main.py                        # answer the demo questions (full RAG)
    python main.py --ask "your question"  # answer one question, show sources
    python main.py --ask "..." --show-chunks   # also print the retrieved chunks
    python main.py --compare "question"   # baseline (no retrieval) vs RAG
    python main.py --retrieval-demo       # Day 4 golden-set retrieval check
    python main.py --evaluate             # Day 6: good/bad question sets + tuning report
    python main.py --reindex | --rebuild  # refresh / rebuild the index
Backends are configured via .env (see .env.example): embeddings + LLM.
"""

import argparse
import logging
import os
import time

try:  # .env support is optional but convenient
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from chunking import split_documents
from embedder import get_embedder
from evaluation import BAD_QUESTIONS, GOOD_QUESTIONS, run_evaluation
from generator import get_generator
from loader import load_knowledge_base
from rag_pipeline import RAGPipeline
from retriever import DEFAULT_TOP_K, Retriever
from schemas import RAGAnswer, RetrievedChunk
from weaviate_store import WeaviateStore

KNOWLEDGE_BASE_DIR = "knowledge_base"

logger = logging.getLogger("rag.main")

# Demo questions come from the Day 6 evaluation sets (single source of
# truth): five answerable ones plus one out-of-scope control.
TEST_QUESTIONS: list[dict] = [
    *GOOD_QUESTIONS[:5],
    {**BAD_QUESTIONS[0], "expect": None},
]

WIDTH = 108


def setup_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------------------------------------------------------------
# Printing helpers
# ----------------------------------------------------------------------

def print_hits(question: str, results: list[RetrievedChunk], mode_label: str) -> None:
    print(f"\nQ: {question}    [{mode_label}, top_k={len(results)}]")
    print(f"   {'rank':<5} {'score':<7} {'distance':<9} {'chunk_id':<19} source_name")
    for r in results:
        distance = f"{r.distance:.3f}" if r.distance is not None else "-"
        print(f"   {r.rank:<5} {r.score:<7.3f} {distance:<9} {r.chunk_id:<19} {r.source_name}")
        print(f"         {r.preview(100)}")


def print_answer(result: RAGAnswer) -> None:
    print("\n" + "-" * WIDTH)
    print(f"Q: {result.question}")
    label = "RAG (retrieval + generation)" if result.used_retrieval else "BASELINE (no retrieval)"
    confidence = (f", retrieval confidence {result.retrieval_confidence:.2f}"
                  if result.retrieval_confidence is not None else "")
    print(f"A [{label}, model: {result.model}{confidence}]:")
    if result.weak_context:
        print("   (weak context detected — refused BEFORE calling the LLM)")
    print(f"   {result.answer}")
    if result.sources:
        print("   Sources:")
        for s in result.sources:
            print(f"     [{s.marker}] {s.chunk_id:<19} {s.source_name} — {s.title}  (score {s.score:.3f})")


# ----------------------------------------------------------------------
# Demo flows
# ----------------------------------------------------------------------

def run_answer_demo(pipeline: RAGPipeline, top_k: int, hybrid: bool, alpha: float) -> None:
    print("\n" + "=" * WIDTH)
    print(f"RAG answer demo — {len(TEST_QUESTIONS)} questions "
          f"(last one is an out-of-scope control)")
    print("=" * WIDTH)
    for item in TEST_QUESTIONS:
        result = pipeline.answer(item["question"], top_k=top_k, hybrid=hybrid, alpha=alpha)
        print_answer(result)


def run_comparison(pipeline: RAGPipeline, question: str, top_k: int) -> None:
    print("\n" + "=" * WIDTH)
    print("Comparison: the same question WITHOUT retrieval vs WITH retrieval")
    print("=" * WIDTH)
    print_answer(pipeline.answer_baseline(question))
    print_answer(pipeline.answer(question, top_k=top_k))
    print("\n   Note: the baseline can only draw on the model's parametric knowledge — it cannot")
    print("   reference these documents and gives no sources; the RAG answer is built from the")
    print("   retrieved fragments and cites them, which makes every claim checkable.")


def run_retrieval_demo(retriever: Retriever, top_k: int, hybrid: bool, alpha: float) -> None:
    """Day 4 golden-set check, kept as a regression tool."""
    mode_label = f"hybrid alpha={alpha}" if hybrid else "vector"
    scored = [q for q in TEST_QUESTIONS if q["expect"]]
    found_count = 0
    print("\n" + "=" * WIDTH)
    print(f"Retrieval demo — {len(TEST_QUESTIONS)} questions, mode: {mode_label}")
    print("=" * WIDTH)
    for item in TEST_QUESTIONS:
        results = retriever.retrieve(item["question"], top_k=top_k, hybrid=hybrid, alpha=alpha)
        print_hits(item["question"], results, mode_label)
        expected = item["expect"]
        if expected:
            found = any(r.source_name in expected for r in results)
            found_count += int(found)
            print(f"   expected {' or '.join(expected)} in top-{top_k}: {'HIT' if found else 'MISS'}")
        elif results:
            print(f"   out-of-scope control — top score {results[0].score:.3f}")
    print("\n" + "-" * WIDTH)
    print(f"Golden-set check: {found_count}/{len(scored)} questions had an expected source in the top-{top_k}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG pipeline — Day 5: grounded answers over the knowledge base"
    )
    parser.add_argument("--ask", metavar="QUESTION", help="answer one question via the full RAG cycle")
    parser.add_argument("--compare", metavar="QUESTION",
                        help="answer one question WITHOUT retrieval and WITH retrieval, side by side")
    parser.add_argument("--show-chunks", action="store_true",
                        help="with --ask: also print the retrieved chunks")
    parser.add_argument("--retrieval-demo", action="store_true",
                        help="run the Day 4 golden-set retrieval check instead of answering")
    parser.add_argument("--evaluate", action="store_true",
                        help="run the Day 6 evaluation: answerable + unanswerable question sets")
    parser.add_argument("--weak-threshold", type=float, default=None,
                        help="override the weak-context gate (default from env or 0.55)")
    parser.add_argument("--min-chunk-score", type=float, default=None,
                        help="override the per-chunk noise filter (default from env or 0.52)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--hybrid", action="store_true", help="use BM25+vector hybrid retrieval")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--reindex", action="store_true",
                        help="re-import chunks even if the collection is populated")
    parser.add_argument("--rebuild", action="store_true",
                        help="drop the collection and re-index")
    args = parser.parse_args()

    setup_logging()

    logger.info("Step 1 — loading documents from '%s/'", KNOWLEDGE_BASE_DIR)
    documents = load_knowledge_base(KNOWLEDGE_BASE_DIR)

    logger.info("Step 2 — splitting documents into chunks")
    chunks = split_documents(documents)

    embedder = get_embedder()

    with WeaviateStore.connect() as store:
        store.ensure_collection(model_label=embedder.model_name, recreate=args.rebuild)

        existing = store.count()
        if args.rebuild or args.reindex or existing == 0:
            logger.info("Step 3 — embedding %d chunks", len(chunks))
            started = time.perf_counter()
            vectors = embedder.embed_texts([chunk.text for chunk in chunks])
            logger.info("Embedded %d chunks with %s in %.2fs (dimension=%d)",
                        len(vectors), embedder.model_name,
                        time.perf_counter() - started, embedder.dimension)
            logger.info("Step 4 — indexing chunks into Weaviate")
            store.index_chunks(chunks, vectors)
        else:
            logger.info("Index already populated (%d objects) — skipping import "
                        "(--reindex / --rebuild to refresh)", existing)

        retriever = Retriever(store, embedder)

        if args.retrieval_demo:
            run_retrieval_demo(retriever, top_k=args.top_k, hybrid=args.hybrid, alpha=args.alpha)
            return

        pipeline = RAGPipeline(
            retriever,
            get_generator(),
            weak_threshold=args.weak_threshold,
            min_chunk_score=args.min_chunk_score,
        )

        if args.evaluate:
            run_evaluation(pipeline, retriever, top_k=args.top_k)
        elif args.ask:
            if args.show_chunks:
                hits = retriever.retrieve(args.ask, top_k=args.top_k,
                                          hybrid=args.hybrid, alpha=args.alpha)
                print_hits(args.ask, hits, "hybrid" if args.hybrid else "vector")
            print_answer(pipeline.answer(args.ask, top_k=args.top_k,
                                         hybrid=args.hybrid, alpha=args.alpha))
        elif args.compare:
            run_comparison(pipeline, args.compare, top_k=args.top_k)
        else:
            run_answer_demo(pipeline, top_k=args.top_k, hybrid=args.hybrid, alpha=args.alpha)
            run_comparison(
                pipeline,
                "Why are documents split into chunks instead of being embedded whole?",
                top_k=args.top_k,
            )

    logger.info("Pipeline with weak-context guards is working — next step: final integration.")


if __name__ == "__main__":
    main()
