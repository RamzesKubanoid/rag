# RAG on Weaviate — Knowledge Base Q&A

An educational, modular RAG (Retrieval-Augmented Generation) service in Python.
Target pipeline: **documents → chunking → embeddings → Weaviate → retrieval → grounded LLM answer with sources**.

The project is built step by step over 7 days. **Current progress: Day 6 — weak-context guards + evaluation.**

## Roadmap

- [x] **Day 1 — Document loading.** `knowledge_base/` folder, loading `.txt`/`.md` files, cleaning, console summary, project skeleton.
- [x] **Day 2 — Chunking.** Paragraph-aware splitting with overlap and per-chunk metadata. Self-checks in `test_chunking.py`.
- [x] **Day 3 — Weaviate + embeddings.** Weaviate via Docker (or Embedded), `KnowledgeChunk` collection, embeddings per chunk (OpenAI or offline fallback), idempotent batch indexing, verification. Self-checks in `test_embedder.py`.
- [x] **Day 4 — Retrieval.** Question normalization, query embedding, Weaviate `near_vector` top-k with score/distance/source, hybrid (BM25+vector) mode, golden-set demo with 6 test questions. Self-checks in `test_retriever.py`.
- [x] **Day 5 — Grounded answers.** Full cycle question → retrieval → context prompt → LLM → answer with [n] citations + source list; strict anti-fabrication prompt; no-retrieval baseline comparison; offline extractive fallback. Self-checks in `test_pipeline.py`.
- [x] **Day 6 — Weak-context handling + evaluation.** Certainty-based noise filter and pre-LLM refusal gate, hard-negative question sets, `--evaluate` report with threshold-tuning guidance. Self-checks in `test_weak_context.py`.
- [ ] Day 7 — final integration & polish.

## Project structure

```
rag-weaviate/
├── main.py               # entry point — full RAG demo / --ask / --compare / --retrieval-demo
├── loader.py             # Day 1: document loading + text cleaning
├── chunking.py           # Day 2: paragraph-aware chunking with overlap
├── embedder.py           # Day 3: OpenAI embeddings + offline hashing fallback
├── weaviate_store.py     # Day 3: connection, schema, idempotent import, verification
├── schemas.py            # pydantic models: TextItem base, Document, Chunk
├── test_chunking.py      # Day 2 self-checks
├── test_embedder.py      # Day 3 self-checks (offline backend)
├── test_retriever.py     # Day 4 self-checks (hit mapping, query normalization)
├── test_pipeline.py      # Day 5 self-checks (prompts, extractive generator, pipeline)
├── test_weak_context.py  # Day 6 self-checks (gate, noise filter, hybrid trap)
├── retriever.py          # Day 4: top-k vector & hybrid retrieval with scores
├── generator.py          # Day 5: OpenAI-compatible LLM + offline extractive fallback
├── prompts.py            # Day 5: grounding rules, context formatting, refusal sentence
├── rag_pipeline.py       # Day 5/6: orchestration + noise filter + weak-context gate
├── evaluation.py         # Day 6: good/bad question sets + qualitative report
├── knowledge_base/       # 8 short articles about RAG concepts (.txt)
├── docker-compose.yml    # local Weaviate 1.38.2 (REST 8080, gRPC 50051)
├── .env.example          # configuration template (keys, backends, ports)
├── requirements.txt
└── README.md
```

## Quickstart (Day 3)

Python 3.10+ required. Docker is optional (see embedded mode below).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # put your OPENAI_API_KEY here (optional)

# 1) Start Weaviate (option A — Docker, recommended)
docker compose up -d

# 2) Index (first run) + retrieval demo over 6 golden-set questions
python main.py

# Ask your own question (full RAG cycle: answer + sources)
python main.py --ask "Why are documents split into chunks?"
python main.py --ask "What does alpha control?" --hybrid --show-chunks

# Compare: the same question answered WITHOUT retrieval vs WITH retrieval
python main.py --compare "Why are documents split into chunks?"

# Day 4 golden-set retrieval check (regression tool)
python main.py --retrieval-demo

# Day 6: evaluate over answerable + unanswerable question sets
python main.py --evaluate
python main.py --evaluate --weak-threshold 0.6   # experiment with the gate

# Index management
python main.py --reindex             # re-import after editing the knowledge base
python main.py --rebuild             # drop + re-index after changing chunking/embedder

# Self-checks
python test_chunking.py
python test_embedder.py
python test_retriever.py
python test_pipeline.py
python test_weak_context.py
```

**Option B — no Docker:** set `WEAVIATE_MODE=embedded` in `.env`. The client downloads a
Weaviate binary on first use and runs it as a subprocess (data persists in `./weaviate_data/`).
Handy for CI and quick experiments; Docker is the primary, production-like path.

## Embeddings (`embedder.py`)

| backend | when | quality |
|---------|------|---------|
| `OpenAIEmbedder` | `OPENAI_API_KEY` set (or `EMBEDDING_BACKEND=openai`) | real semantic vectors (`text-embedding-3-small`, 1536-d by default) |
| `HashingEmbedder` | no key / `EMBEDDING_BACKEND=hash` | offline, deterministic, keyword-overlap only — for free development & tests |

`OPENAI_BASE_URL` switches to any OpenAI-compatible provider, and each component can also use
its own provider via `EMBEDDING_API_KEY`/`EMBEDDING_BASE_URL` and `LLM_API_KEY`/`LLM_BASE_URL`
(falling back to the shared `OPENAI_*` vars). Typical mixed setup: the LLM from Hugging Face
Inference Providers (`LLM_BASE_URL=https://router.huggingface.co/v1`, model e.g.
`openai/gpt-oss-20b`, HF token with the "Make calls to Inference Providers" permission — note
HF's OpenAI-compatible endpoint serves chat only, not embeddings) plus Gemini or the offline
hash backend for embeddings. Documents and queries are always embedded by the same backend; the collection remembers which embedder built it (in its
description) and the store logs a warning if you later connect with a different one — in that
case re-index with `--rebuild`, since vectors from different models are incompatible.

## Weaviate storage (`weaviate_store.py`)

* Collection `KnowledgeChunk`, **bring-your-own-vectors** (no vectorizer module), HNSW index
  with **cosine** distance.
* Properties: `chunk_id`, `document_id`, `source_name`, `title`, `chunk_index`, `text`.
* **Idempotent indexing:** every object's UUID is `uuid5(chunk_id)`, so re-importing the same
  knowledge base *overwrites* objects instead of duplicating them (verified: two consecutive
  runs keep the count at 24).
* **Caveat:** if you change chunking parameters or the embedding model, the chunk ids / vectors
  change — old objects would linger. That is exactly what `python main.py --rebuild` is for.
* Verification: `main.py` prints the object count (expected vs actual) and two sample objects
  with all metadata and the stored vector dimension.

## How chunking works (Day 2, `chunking.py`)

Paragraph-aware packing with a sentence-level fallback: text splits into paragraphs (Day 1
preserved blank-line boundaries); oversized paragraphs split into sentences; monster sentences
hard-split by words. New content per chunk ≤ `max_chars` (default 1000 ≈ 150–200 words); each
next chunk starts with the last sentences of the previous one (`overlap_chars=200`, ~20%), so
boundary thoughts survive. Tiny trailing chunks merge back when they fit (`min_chunk_chars=200`).
Full chunk text ≤ `max_chars + overlap_chars + 2`.

## What cleaning does (Day 1, `loader.clean_text`)

Unicode NFKC normalization; CRLF/CR → LF; tabs & non-breaking spaces → spaces; collapsed space
runs; per-line trim; multiple blank lines → one. A single blank line is deliberately **kept** as
the paragraph separator the chunker relies on.

## Knowledge base

`knowledge_base/` ships with 8 articles about RAG itself — embeddings, vector databases,
Weaviate, chunking, prompting, hallucinations, evaluation — so the finished system can answer
questions about how it works from its own documents. Files `02` and `05` intentionally contain
messy whitespace so the cleaning step visibly does something. Any UTF-8 `.txt`/`.md` dropped
into `knowledge_base/` is picked up automatically.

## Retrieval (Day 4, `retriever.py`)

Flow: normalize the question → embed it **with the same backend that indexed the chunks** →
Weaviate `near_vector` (or `hybrid`) → top-k `RetrievedChunk` objects with `rank`, `score`,
`distance`, `chunk_id`, `source_name`, `text`.

Score semantics: in vector mode `score` is Weaviate's *certainty* for cosine (0..1, higher =
closer; 0.5 means orthogonal, i.e. no similarity at all) and `distance` is the raw cosine
distance. In hybrid mode `score` is the fused BM25+vector value (`alpha`: 0 = keywords only,
1 = vectors only). Beware: fusion normalizes scores **per query** (relative score fusion), so
the top hit of *any* query scores near the top of the scale — never use the fused score as an
absolute confidence threshold. Since the hybrid API returns no geometric measure, the store
also computes each hit's true cosine `distance` client-side from its stored vector, so you can
see both signals (a hit with distance 1.0 got in purely via keywords). Hybrid requires passing
the query vector explicitly because the collection brings its own vectors.

`python main.py` (no args) skips re-indexing when the collection is already populated and runs
a golden-set demo: 5 in-scope questions annotated with the source file expected in the top-k
(printed as HIT/MISS) plus one out-of-scope control ("What is the capital of France?") whose
visibly lower top score previews the Day 6 weak-context threshold. It finishes with a
vector-vs-hybrid side-by-side where BM25 promotes the chunk containing the literal query term.

Note on the offline embedder: it now filters ~60 English stopwords (`hashing-bow-…-v2`) —
without that, function words dominated and even unrelated questions scored high. If you have an
index built by the Day 3 version, the store will warn about the embedder mismatch: run
`python main.py --rebuild` once. With a real embedding API the in-scope/out-of-scope score gap
becomes far larger than the hash backend can show.

## Grounded generation (Day 5, `prompts.py` + `generator.py` + `rag_pipeline.py`)

Full cycle: `RAGPipeline.answer(question)` → retrieve top-k chunks → format them as numbered,
source-labelled fragments → send with a strict system prompt → return a `RAGAnswer` (pydantic)
holding the answer text, a source list mirroring the [n] markers, the retrieved chunks and the
model name.

The system prompt is the anti-hallucination contract: answer ONLY from the numbered fragments,
never invent facts beyond them, cite every claim with [n], and — if the fragments are not
enough — reply with the exact refusal sentence (`prompts.NO_ANSWER_SENTENCE`), which the
pipeline and tests can recognize. Generation runs at low temperature by default (0.2).

Generation backends (mirroring the embedder design):

| backend | when | quality |
|---------|------|---------|
| `OpenAIGenerator` | `OPENAI_API_KEY` set (or `LLM_BACKEND=openai`) | real LLM via OpenAI or any compatible endpoint (Gemini, Anthropic, Ollama, ... through `OPENAI_BASE_URL`); model via `LLM_MODEL` |
| `ExtractiveGenerator` | no key / `LLM_BACKEND=extractive` | offline stand-in: copies the sentences most lexically relevant to the question and cites their chunks; declines when nothing overlaps. Cannot rephrase or synthesize — for pipeline development only |

`python main.py --compare "..."` answers the same question twice: the baseline (no retrieval,
parametric knowledge only, no sources) and the RAG answer (built from fragments, every claim
citable). With the offline fallback the baseline honestly reports it has no knowledge source —
which is itself the point of the comparison; with a real LLM you will typically see a plausible
but generic and unverifiable baseline vs a specific, cited RAG answer.

## Weak-context handling & evaluation (Day 6, `rag_pipeline.py` + `evaluation.py`)

Three defense layers against confident hallucination, in execution order:

1. **Noise filter** (`RETRIEVAL_MIN_CHUNK_SCORE`, default 0.52) — retrieved chunks whose
   certainty is below the bar never enter the prompt. Targets obvious junk (orthogonal 0.50
   hits), not borderline chunks, since mildly-noisy context is the prompt's job to handle.
2. **Confidence gate** (`RETRIEVAL_WEAK_THRESHOLD`, default 0.55) — if the *best* certainty is
   below the gate, the pipeline returns the honest refusal sentence immediately: no LLM call,
   no tokens, no hallucination risk. `RAGAnswer.weak_context=True` marks these.
3. **Prompt rules** — questions on-topic enough to pass the gate but unanswered by the
   fragments (hard negatives, e.g. "How much does Weaviate Cloud cost?") can only be refused
   by the model itself. This layer needs a real LLM; the extractive fallback fails hard
   negatives by design.

Gating always uses **certainty** (`1 − cosine_distance/2`), never the hybrid fused score —
fusion is normalized per query, so even junk queries produce a top hit near the top of the
scale (guarded by a dedicated test).

**Measured with the offline hash embedder** (`python main.py --evaluate`, 8 answerable + 6
unanswerable questions): answerable — 8/8 expected sources in top-4 (6 at rank 1), all
answered, confidence 0.58–0.75; unanswerable — confidence 0.53–0.63, gate caught 2, extractive
generator refused 1, and 3 leaked (one borderline off-topic + two hard negatives). The key
finding: with the hash embedder the GOOD and BAD confidence ranges **overlap** (0.58–0.63), so
no threshold alone can separate them — that is precisely why the defense is layered, and why
real embeddings matter: they widen the gap. After switching embedders, re-run `--evaluate` and
set the gate inside the reported gap.

**Settings that work for this KB** (documented per Day 6 task): chunking `max_chars=1000` /
`overlap=200`, paragraph-aware (24 chunks, no mid-thought cuts, 2–3 chunks per article);
`top_k=4` (enough coverage, little noise); gate 0.55 / filter 0.52 for `hashing-bow-…-v2`.
Keep the hard negatives in the evaluation set permanently — they are the regression test for
the whole anti-hallucination stack.
