"""Day 5 — answer generation backends.

Mirrors the embedder design: one interface, two implementations.

  * OpenAIGenerator    — a real LLM through the OpenAI API or ANY
    OpenAI-compatible endpoint (OPENAI_BASE_URL): OpenAI, Gemini,
    Anthropic, Ollama, ... Used when OPENAI_API_KEY is set.
  * ExtractiveGenerator — offline stand-in that copies the most
    question-relevant sentences from the retrieved chunks, with [n]
    citations. No key, no cost, deterministic; lets the full RAG cycle
    run and be tested. It cannot rephrase or synthesize — switch to a
    real LLM for actual generation quality.
"""

import logging
import os
import re
from abc import ABC, abstractmethod

from chunking import _split_sentences
from embedder import STOPWORDS
from prompts import (
    NO_ANSWER_SENTENCE,
    SYSTEM_PROMPT_BASELINE,
    SYSTEM_PROMPT_RAG,
    build_rag_user_prompt,
)
from schemas import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = "gpt-4o-mini"

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(text.lower())
        if len(token) >= 2 and token not in STOPWORDS
    }


class Generator(ABC):
    """Common interface: produce answers with and without retrieved context."""

    model_name: str

    @abstractmethod
    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Grounded answer built ONLY from the given chunks, with [n] citations."""

    @abstractmethod
    def answer_baseline(self, question: str) -> str:
        """No-retrieval answer from the model's own knowledge (for comparison)."""


class OpenAIGenerator(Generator):
    """LLM answers via the OpenAI API or any compatible endpoint."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        from openai import OpenAI  # local import keeps the dependency optional

        self.model_name = model or os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        self.temperature = (
            float(os.getenv("LLM_TEMPERATURE", "0.2")) if temperature is None else temperature
        )
        self.max_tokens = (
            int(os.getenv("LLM_MAX_TOKENS", "800")) if max_tokens is None else max_tokens
        )
        # Per-component overrides first, shared OPENAI_* variables as fallback —
        # e.g. LLM_BASE_URL=https://router.huggingface.co/v1 serves the LLM from
        # Hugging Face while embeddings come from another provider.
        self._client = OpenAI(
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None,
        )

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> str:
        return self._complete(SYSTEM_PROMPT_RAG, build_rag_user_prompt(question, chunks))

    def answer_baseline(self, question: str) -> str:
        return self._complete(SYSTEM_PROMPT_BASELINE, question)


class ExtractiveGenerator(Generator):
    """Offline fallback: extract the most relevant sentences, cite their chunks.

    'Relevance' is content-word overlap with the question (same stopword
    filtering as the offline embedder). Sentences repeated via chunk
    overlap are deduplicated, keeping the earliest citation marker.
    """

    model_name = "extractive-fallback"
    _MAX_SENTENCES = 3

    def answer_with_context(self, question: str, chunks: list[RetrievedChunk]) -> str:
        question_tokens = _content_tokens(question)
        seen: set[str] = set()
        scored: list[tuple[int, int, int, str]] = []  # (overlap, marker, order, sentence)
        order = 0
        for marker, chunk in enumerate(chunks, start=1):
            for sentence in _split_sentences(chunk.text):
                key = " ".join(sentence.lower().split())
                if key in seen:
                    continue  # overlap tails repeat sentences across chunks
                seen.add(key)
                overlap = len(question_tokens & _content_tokens(sentence))
                if overlap:
                    scored.append((overlap, marker, order, sentence))
                    order += 1

        if not scored:
            return NO_ANSWER_SENTENCE

        best = sorted(scored, key=lambda item: -item[0])[: self._MAX_SENTENCES]
        best.sort(key=lambda item: (item[1], item[2]))  # readable order: by chunk, then position
        return " ".join(f"{sentence} [{marker}]" for _, marker, _, sentence in best)

    def answer_baseline(self, question: str) -> str:
        return (
            "(offline fallback) Without retrieval there is no knowledge source to draw "
            "from, so no answer can be given — the no-retrieval baseline requires a real LLM."
        )


def get_generator() -> Generator:
    """Pick the generation backend from environment configuration.

    LLM_BACKEND = auto (default) | openai | extractive
      auto       -> OpenAI-compatible LLM when OPENAI_API_KEY is set,
                    otherwise the offline extractive fallback.
    LLM_MODEL sets the model name (default gpt-4o-mini).
    """
    backend = os.getenv("LLM_BACKEND", "auto").lower()
    if backend not in {"auto", "openai", "extractive"}:
        raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (use auto|openai|extractive)")

    has_key = bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if backend == "openai" or (backend == "auto" and has_key):
        generator = OpenAIGenerator()
        logger.info("Generation backend: OpenAI-compatible LLM (%s)", generator.model_name)
        return generator

    if backend == "auto":
        logger.warning(
            "No LLM_API_KEY / OPENAI_API_KEY set — falling back to the offline ExtractiveGenerator. "
            "Answers will be copied sentences, not real generation; set the key "
            "(see .env.example) to use an LLM."
        )
    else:
        logger.info("Generation backend: offline ExtractiveGenerator")
    return ExtractiveGenerator()
