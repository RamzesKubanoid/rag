"""Evaluation question sets and a qualitative retrieval/answer report.

Two question groups:
  * GOOD_QUESTIONS — the knowledge base contains the answer; each entry
    names the source file(s) expected among the top-k hits.
  * BAD_QUESTIONS — the knowledge base must NOT answer; includes easy
    off-topic cases (caught by the confidence gate) and HARD NEGATIVES:
    questions that share vocabulary or entities with the KB but whose
    answer is absent — those pass the gate and must be refused by the
    generator layer (the strict prompt). The offline extractive fallback
    cannot judge answerability semantically, so hard negatives are
    expected to fail with it; a real LLM should refuse them.

`run_evaluation` answers every question through the full pipeline and
prints per-question rows plus a summary with threshold-tuning guidance.
"""

import logging

from prompts import NO_ANSWER_SENTENCE
from rag_pipeline import RAGPipeline, certainty
from retriever import Retriever
from schemas import RAGAnswer

logger = logging.getLogger(__name__)

GOOD_QUESTIONS: list[dict] = [
    {"question": "Why are documents split into chunks instead of being embedded whole?",
     "expect": ["05_chunking_strategies.txt"]},
    {"question": "What is cosine similarity used for when comparing embeddings?",
     "expect": ["02_text_embeddings.txt"]},
    {"question": "How does hybrid search combine keyword and vector signals?",
     "expect": ["04_weaviate_basics.txt", "03_vector_databases.txt"]},
    {"question": "What is HNSW and why do vector databases use approximate search?",
     "expect": ["03_vector_databases.txt"]},
    {"question": "What are the two ways to get vectors into Weaviate?",
     "expect": ["04_weaviate_basics.txt"]},
    {"question": "What is groundedness and how can it be checked?",
     "expect": ["07_hallucinations_and_grounding.txt"]},
    {"question": "Which metrics measure retrieval quality in a RAG system?",
     "expect": ["08_evaluating_rag.txt"]},
    {"question": "What should the model do when the context does not contain the answer?",
     "expect": ["06_prompting_with_context.txt"]},
]

BAD_QUESTIONS: list[dict] = [
    {"question": "What is the capital of France?",
     "why": "general world knowledge, lexically unrelated to the KB"},
    {"question": "How do I bake sourdough bread at home?",
     "why": "unrelated domain"},
    {"question": "Who won the FIFA World Cup in 2022?",
     "why": "unrelated domain, time-sensitive fact"},
    {"question": "What is the average salary of a data scientist?",
     "why": "off-topic; only the word 'data' overlaps"},
    {"question": "How do I set up PostgreSQL streaming replication?",
     "why": "HARD NEGATIVE: database vocabulary overlaps the KB (pgvector is mentioned)"},
    {"question": "How much does a Weaviate Cloud subscription cost?",
     "why": "HARD NEGATIVE: entity exists in the KB, the fact does not — "
            "must be refused by the generator layer, not the gate"},
]


def _behavior(result: RAGAnswer) -> str:
    """Classify how the pipeline handled a question."""
    if result.weak_context:
        return "REFUSED(gate)"
    if NO_ANSWER_SENTENCE in result.answer:
        return "REFUSED(generator)"
    return "ANSWERED"


def run_evaluation(pipeline: RAGPipeline, retriever: Retriever, top_k: int) -> None:
    """Qualitative evaluation of retrieval + end-to-end behavior."""
    width = 108
    print("\n" + "=" * width)
    print(f"Evaluation — {len(GOOD_QUESTIONS)} answerable + {len(BAD_QUESTIONS)} unanswerable "
          f"questions (top_k={top_k}, gate={pipeline.weak_threshold:.2f}, "
          f"noise filter={pipeline.min_chunk_score:.2f})")
    print("=" * width)

    # ---- answerable ---------------------------------------------------
    print("\nAnswerable questions (expected source should be retrieved; pipeline should ANSWER):")
    print(f"  {'hit':<6} {'conf':<6} {'kept':<6} {'behavior':<19} question")
    good_conf: list[float] = []
    good_hits = good_answered = 0
    for item in GOOD_QUESTIONS:
        hits = retriever.retrieve(item["question"], top_k=top_k)
        rank = next((h.rank for h in hits if h.source_name in item["expect"]), None)
        kept = sum(1 for h in hits if certainty(h) >= pipeline.min_chunk_score)
        result = pipeline.answer(item["question"], top_k=top_k)
        behavior = _behavior(result)
        good_conf.append(result.retrieval_confidence or 0.0)
        good_hits += rank is not None
        good_answered += behavior == "ANSWERED"
        hit_label = f"@{rank}" if rank else "MISS"
        print(f"  {hit_label:<6} {result.retrieval_confidence:<6.3f} {kept}/{len(hits):<4} "
              f"{behavior:<19} {item['question']}")

    # ---- unanswerable --------------------------------------------------
    print("\nUnanswerable questions (pipeline should REFUSE — via the gate or the generator):")
    print(f"  {'conf':<6} {'kept':<6} {'behavior':<19} question")
    bad_conf: list[float] = []
    refused_gate = refused_generator = wrongly_answered = 0
    for item in BAD_QUESTIONS:
        hits = retriever.retrieve(item["question"], top_k=top_k)
        kept = sum(1 for h in hits if certainty(h) >= pipeline.min_chunk_score)
        result = pipeline.answer(item["question"], top_k=top_k)
        behavior = _behavior(result)
        bad_conf.append(result.retrieval_confidence or 0.0)
        refused_gate += behavior == "REFUSED(gate)"
        refused_generator += behavior == "REFUSED(generator)"
        wrongly_answered += behavior == "ANSWERED"
        flag = "  <-- FAILURE" if behavior == "ANSWERED" else ""
        print(f"  {result.retrieval_confidence:<6.3f} {kept}/{len(hits):<4} "
              f"{behavior:<19} {item['question']}{flag}")
        print(f"         ({item['why']})")

    # ---- summary --------------------------------------------------------
    refused = refused_gate + refused_generator
    print("\n" + "-" * width)
    print(f"GOOD: expected source in top-{top_k}: {good_hits}/{len(GOOD_QUESTIONS)}   "
          f"answered: {good_answered}/{len(GOOD_QUESTIONS)}   "
          f"confidence range {min(good_conf):.3f}-{max(good_conf):.3f}")
    print(f"BAD : refused: {refused}/{len(BAD_QUESTIONS)} "
          f"(gate {refused_gate}, generator {refused_generator})   "
          f"wrongly answered: {wrongly_answered}   "
          f"confidence range {min(bad_conf):.3f}-{max(bad_conf):.3f}")
    print(f"Thresholds: gate={pipeline.weak_threshold:.2f}, "
          f"noise filter={pipeline.min_chunk_score:.2f} "
          "(env: RETRIEVAL_WEAK_THRESHOLD / RETRIEVAL_MIN_CHUNK_SCORE)")
    print("Tuning: after changing the embedder, re-run this report and place the gate "
          "threshold inside the gap\nbetween the GOOD and BAD confidence ranges. "
          "Hard negatives cannot be caught by any threshold —\nthey rely on the LLM "
          "honoring the prompt rules (the extractive fallback fails them by design).")
    print("-" * width)
