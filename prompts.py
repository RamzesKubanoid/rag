"""Prompt templates for grounded answer generation.

The system prompt is the main anti-hallucination tool of the pipeline:
it forbids outside knowledge, demands [n] citations for every claim and
defines the exact refusal sentence for insufficient context (Day 6 will
also detect that case *before* calling the LLM).
"""

from schemas import RetrievedChunk

# The exact sentence the model must use when the context is not enough.
# Keeping it as a constant lets the pipeline and tests recognize it.
NO_ANSWER_SENTENCE = (
    "The knowledge base does not contain enough information to answer this question."
)

SYSTEM_PROMPT_RAG = f"""You are the answer engine of a small RAG system built over a local knowledge base.

Strict rules:
1. Use ONLY the numbered context fragments provided in the user message. Do not use any outside or prior knowledge, even if you are confident about it.
2. Never invent, assume or extrapolate facts that are not stated in the fragments.
3. Cite the fragment(s) behind every claim with markers like [1] or [2][3] placed right after the claim.
4. If the fragments do not contain the information needed, reply with exactly this sentence: "{NO_ANSWER_SENTENCE}" You may add one sentence describing what related information the fragments do contain.
5. Be concise: 2-6 sentences, no preamble, no repetition of the question."""

SYSTEM_PROMPT_BASELINE = (
    "You are a helpful assistant. Answer the question concisely (2-6 sentences) "
    "from your own general knowledge."
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the fragments and label each with its origin, e.g.:

    [1] (source: 05_chunking_strategies.txt, chunk: doc_005_chunk_000)
    <chunk text>
    """
    blocks = []
    for marker, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{marker}] (source: {chunk.source_name}, chunk: {chunk.chunk_id})\n{chunk.text}"
        )
    return "\n\n".join(blocks)


def build_rag_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "Context fragments:\n\n"
        f"{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the fragments above, citing them with [n] markers."
    )
