# RAG on Weaviate — Knowledge Base Q&A

Modular Retrieval-Augmented Generation mini-product in Python.

```
documents → clean → chunk → embed ┐
                                  ├→ Weaviate (HNSW, cosine)
question  → normalize → embed ────┘        │
                                     top-k chunks + certainty
                                           │
                        noise filter → weak-context gate ──(too weak)──→ honest refusal
                                           │
                        numbered context + strict prompt → LLM
                                           │
                        grounded answer + [n] citations + sources
```

**What it does:** loads local documents, splits them into overlapping chunks, stores chunks +
embeddings + metadata in Weaviate, retrieves relevant context for a question, generates an
answer grounded ONLY in that context, cites its sources, and refuses honestly when retrieval
is weak instead of hallucinating. Usable from a CLI and over an HTTP API.

---

## Quickstart

Python 3.10+. Works fully offline out of the box (deterministic fallback backends); add one
API key for real semantic search and real generated answers.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # optional: add your API key(s)
```

### 1. Spin up Weaviate

```bash
docker compose up -d          # Weaviate 1.38.2 on :8080 (REST) and :50051 (gRPC)
docker compose down           # stop (data persists);  down -v  wipes the data
```

No Docker? Set `WEAVIATE_MODE=embedded` in `.env` — the client downloads a Weaviate binary
and runs it as a subprocess (data in `./weaviate_data/`).

### 2. Prepare the knowledge base

Drop UTF-8 `.txt` / `.md` files into `knowledge_base/`. The repo ships with 8 articles about
RAG itself (embeddings, vector DBs, Weaviate, chunking, prompting, hallucinations,
evaluation), so the system can answer questions about how it works from its own documents.
Replace them with your own at any time, then `python main.py --reindex`.

### 3. Load chunks into Weaviate

```bash
python main.py                # first run indexes automatically, then runs the demo
python main.py --reindex      # re-import after editing the knowledge base
python main.py --rebuild      # drop + re-index (after changing chunking or the embedder)
```

Indexing is idempotent: object UUIDs are `uuid5(chunk_id)`, so re-runs upsert instead of
duplicating. Startup also runs a vector-index health probe and rebuilds automatically if the
index was corrupted by an unclean shutdown.

### 4. Ask questions

```bash
python main.py --ask "Why are documents split into chunks?"
python main.py --ask "What does alpha control?" --hybrid --show-chunks
python main.py --compare "Why are documents split into chunks?"   # baseline vs RAG
python main.py --evaluate                                          # quality report
```

Or over HTTP:

```bash
uvicorn api:app               # or: python api.py   → interactive docs at /docs
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question":"What is groundedness and how can it be checked?","top_k":4}'
curl -s -X POST localhost:8000/search -H 'content-type: application/json' \
     -d '{"question":"hybrid search alpha","top_k":3}'              # retrieval only
```

`POST /ask` accepts `{question, top_k?, hybrid?, alpha?}` and returns the full `RAGAnswer`.
A weak-context refusal is a normal `200` response with `"weak_context": true` and an empty
source list — a refusal is a correct answer, not an error.

### 5. Interpret the sources

Example answer (real run: `openai/gpt-oss-20b` via Hugging Face, offline hash embeddings):

```
Q: What is groundedness and how can it be checked?
A [RAG (retrieval + generation), model: openai/gpt-oss-20b, retrieval confidence 0.58]:
   Groundedness is the property that every claim in an answer is supported by an explicitly
   provided source. It can be checked by taking each sentence of the answer and verifying
   that some retrieved fragment supports it. [1]
   Sources:
     [1] doc_007_chunk_001   07_hallucinations_and_grounding.txt — Hallucinations and ...  (score 0.583)
     [2] doc_007_chunk_002   07_hallucinations_and_grounding.txt — Hallucinations and ...  (score 0.546)
```

* `[n]` markers in the answer point into the numbered **Sources** list; `source_name` is the
  file in `knowledge_base/`, `chunk_id` locates the exact fragment — open the file to verify
  any claim. Sources list everything the model was given; the markers show what it used.
* `retrieval confidence` is the best chunk certainty: `1 − cosine_distance/2`
  (1.0 identical, 0.5 orthogonal/no similarity). It is the gate's signal.
* Three refusal layers: `weak_context: true` → the confidence **gate** refused before any
  LLM call; the exact sentence "The knowledge base does not contain enough information…"
  without the flag → the **generator** refused per the prompt rules; anything else is a
  grounded, cited answer.

---

## Demo questions

Answerable (each names its expected source; see `evaluation.py`):
1. Why are documents split into chunks instead of being embedded whole?
2. What is cosine similarity used for when comparing embeddings?
3. How does hybrid search combine keyword and vector signals?
4. What is HNSW and why do vector databases use approximate search?
5. What are the two ways to get vectors into Weaviate?
6. What is groundedness and how can it be checked?
7. Which metrics measure retrieval quality in a RAG system?
8. What should the model do when the context does not contain the answer?

Unanswerable controls (must be refused): capital of France, sourdough recipe, FIFA 2022,
data-scientist salary, PostgreSQL replication (hard negative: overlapping vocabulary),
Weaviate Cloud pricing (hard negative: entity in the KB, fact absent).
`python main.py` demos six of these; `--evaluate` runs all fourteen with a report.

## Configuration & providers

Everything is configured in `.env` (see `.env.example`). Both the embedder and the LLM speak
the OpenAI protocol, so any compatible provider works; each component can use its own
provider (`EMBEDDING_API_KEY/_BASE_URL`, `LLM_API_KEY/_BASE_URL`) with shared `OPENAI_*` as
fallback.

| setup | embeddings | LLM | gate (`RETRIEVAL_WEAK_THRESHOLD`) |
|---|---|---|---|
| offline (default, no keys) | `hashing-bow-…-v2` | extractive fallback | 0.55 (measured: good 0.58–0.75 vs bad 0.53–0.63 — ranges overlap, see Evaluation) |
| Google Gemini (tested) | `gemini-embedding-2` | `gemini-2.5-flash` / `gemini-3.1-flash-lite` | **0.80** (measured: good 0.86–0.93 vs thresholdable bad ≤ 0.785) |
| Hugging Face (chat only) | pair with Gemini or hash | `openai/gpt-oss-20b` via `https://router.huggingface.co/v1` | per embedder |
| OpenAI | `text-embedding-3-small` | `gpt-4o-mini` | tune via `--evaluate` |
| Ollama (local) | `nomic-embed-text` | e.g. `llama3.2` | tune via `--evaluate` |

After ANY embedder change run `python main.py --rebuild` — vectors from different models are
incompatible, and the store warns if the collection was built by another embedder.

## How it works (module map)

`loader.py` cleans text but keeps single blank lines — paragraph boundaries the chunker needs
(Day 1). `chunking.py` packs whole paragraphs up to `max_chars=1000` with a 200-char sentence
overlap, falling back to sentence/word splitting for oversized input (Day 2). `embedder.py`
and `weaviate_store.py` embed chunks and store them (bring-your-own-vectors, HNSW + cosine,
idempotent upserts) (Day 3). `retriever.py` returns top-k `RetrievedChunk`s with certainty
and distance for vector or hybrid search — hybrid distance is computed client-side because
the fused score is per-query-normalized and unusable as confidence (Day 4). `prompts.py` +
`generator.py` + `rag_pipeline.py` build the numbered-fragment prompt, call the LLM (or the
offline extractive fallback) and return a `RAGAnswer` with sources (Day 5). The pipeline's
noise filter and confidence gate refuse weak retrieval before the LLM; `evaluation.py`
measures all of it (Day 6). `indexing.py` + `api.py` share the index flow between the CLI and
FastAPI (Day 7). All data contracts are pydantic models in `schemas.py`.

## Evaluation & tuning

`python main.py --evaluate` answers 8 good + 6 bad questions and reports: expected-source
rank, retrieval confidence, kept-chunk ratio, and behavior (ANSWERED / REFUSED(gate) /
REFUSED(generator)), plus the confidence ranges of both groups. Tuning procedure: put the
gate inside the gap between those ranges. Measured results: with the offline hash embedder
the ranges overlap (no threshold separates them — hence the layered defense); with
`gemini-embedding-2` + `gemini-3.1-flash-lite` the system scored 8/8 retrieval hits at rank 1
and refused 6/6 unanswerable questions, and the gap (0.785 vs 0.860) supports a 0.80 gate.
Hard negatives can never be caught by a threshold — the pricing question scores 0.87, inside
the good range — they are refused by the prompt layer, which needs a real LLM.

## Testing

Seven suites, 42 self-checks, no server or key required — run each directly
(`python test_chunking.py` … `python test_api.py`) or all via `python -m pytest -q`.
They cover chunk size/overlap invariants, embedder determinism and provider quirks (Gemini's
null `index` fields), retrieval mapping, prompt contracts, the extractive generator, the
weak-context guards (including the hybrid fused-score trap), the indexing health probe, and
the API contract.

## Troubleshooting

* **Everything is refused with confidence 0.00 although the index has objects** — the vector
  index was corrupted by an unclean shutdown; startup detects and rebuilds it automatically
  (re-importing alone does not repair a corrupted HNSW). Manual fix: `python main.py --rebuild`.
* **"Collection was built with a different embedder" warning** — run `--rebuild`.
* **HTTP 503/429 from a provider** — the OpenAI SDK auto-retries transient errors; on free
  tiers just re-run.

## Ideas for the next module

Re-ranking retrieved chunks with a cross-encoder; streaming answers over the API; an
ingestion endpoint + PDF/DOCX loaders; automated groundedness scoring (LLM-as-judge) on top
of `evaluation.py`; caching query embeddings; auth and rate limiting on the API; packaging
the app itself into docker-compose next to Weaviate.

## Project structure

```
rag-weaviate/
├── main.py               # CLI: demos, --ask/--compare, --evaluate, index management
├── api.py                # FastAPI service: GET /health, POST /search, POST /ask
├── indexing.py           # shared flow: load → chunk → embed → import (+ health probe)
├── loader.py             # document loading + cleaning
├── chunking.py           # paragraph-aware chunking + overlap
├── embedder.py           # embeddings: OpenAI-compatible + hash
├── weaviate_store.py     # schema, idempotent import, search
├── retriever.py          # top-k retrieval with certainty
├── prompts.py            # grounding rules + refusal sentence
├── generator.py          # LLM + offline extractive fallback
├── rag_pipeline.py       # orchestration + weak-context guards
├── evaluation.py         # question sets + quality report
├── schemas.py            # pydantic data contracts
├── knowledge_base/       # 8 sample articles (replace with your own)
├── test_*.py             # 7 suites / 42 checks
├── docker-compose.yml    # Weaviate 1.38.2
├── .env.example          # configuration template
└── requirements.txt
```
