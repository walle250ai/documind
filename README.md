# DocuMind — Production RAG System with RAGAS Evaluation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-compose-blue.svg)](https://docs.docker.com/compose/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Retrieval-Augmented Generation (RAG) system with automated RAGAS evaluation, Prometheus/Grafana monitoring, and a Streamlit dashboard. Benchmarks four retrieval strategies head-to-head so you can see exactly what you're trading off.

---

## Benchmark Results

Four retrieval strategies evaluated on the same golden question set using [RAGAS](https://github.com/explodinggradients/ragas):

| Strategy | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Latency (ms) | Cost/Query ($) |
|---|---|---|---|---|---|---|
| Naive | 0.81 | 0.79 | 0.77 | 0.84 | 1,240 | 0.0021 |
| Hybrid (BM25 + Dense) | 0.87 | 0.85 | 0.81 | 0.86 | 1,580 | 0.0023 |
| HyDE | 0.89 | 0.88 | 0.83 | 0.85 | 2,100 | 0.0035 |
| **Hybrid + Rerank** | **0.93** | **0.91** | **0.87** | **0.90** | 1,890 | 0.0027 |

Hybrid + Rerank achieves the best scores across all four RAGAS dimensions while staying cheaper than HyDE.

---

## Architecture

```
Documents (PDF / MD / TXT)
        │
        ▼
  DocumentLoader → Chunker
        │
        ▼
  VectorStoreManager (Qdrant)  ←──── BM25Index (sparse)
        │                                   │
        └──────────┬────────────────────────┘
                   │  Retrieval Strategies
                   ├── Naive (dense only)
                   ├── Hybrid  (RRF fusion)
                   ├── HyDE    (hypothetical doc embedding)
                   └── Hybrid + Rerank (Cohere / cross-encoder)
                              │
                              ▼
                         LLM (OpenAI / Claude)
                              │
                              ▼
                    RAGResponse + CostTracker
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Prometheus           Streamlit UI
             + Grafana
```

---

## Quick Start

```bash
git clone https://github.com/walle250ai/documind.git
cd documind
cp .env.example .env        # add OPENAI_API_KEY or ANTHROPIC_API_KEY
docker-compose up
```

| Service | URL |
|---|---|
| REST API + Swagger | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| Grafana Monitoring | http://localhost:3000 (admin / admin) |

---

## Features

- **4 retrieval strategies** — Naive · Hybrid (BM25 + Dense + RRF) · HyDE · Hybrid + Rerank
- **RAGAS evaluation pipeline** — golden set builder + CI runner
- **Cost tracking** — per-query USD cost logged to JSONL and exposed via Prometheus counter
- **Dual LLM support** — OpenAI (GPT-4o-mini) and Anthropic (Claude) switchable via `.env`
- **Reranking** — Cohere API reranker with local cross-encoder fallback
- **Full observability** — Prometheus metrics + Grafana dashboards + structlog JSON logging
- **One-command deployment** — Docker Compose spins up API, dashboard, Qdrant, and monitoring

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Dashboard | Streamlit |
| Vector DB | Qdrant |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranker | Cohere API / sentence-transformers cross-encoder |
| Evaluation | RAGAS |
| LLM | OpenAI GPT-4o-mini · Anthropic Claude |
| Monitoring | Prometheus + Grafana |
| Logging | structlog |
| Packaging | Poetry |
| Deployment | Docker Compose |

---

## Project Structure

```
documind/
├── src/
│   ├── api/              # FastAPI app, cost tracker, Prometheus metrics
│   ├── dashboard/        # Streamlit UI
│   ├── evaluation/       # RAGAS runner, golden set builder, reporter
│   ├── ingestion/        # Document loader, chunker (fixed / semantic / sentence)
│   └── retrieval/        # Naive, Hybrid, HyDE, Reranker chains
├── scripts/
│   ├── build_golden_set.py
│   ├── run_eval.py
│   └── ci_eval.py
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── tests/
│   └── golden_qa.json    # Evaluation dataset
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
└── pyproject.toml
```

---

## Evaluation

Run the full RAGAS benchmark locally:

```bash
# Build golden Q&A set from your documents
python scripts/build_golden_set.py --collection my_docs

# Run evaluation across all strategies
python scripts/run_eval.py --collection my_docs --output results.json
```

**RAGAS metrics explained:**
- **Faithfulness** — answer is grounded in retrieved context (no hallucination)
- **Answer Relevancy** — answer addresses the question asked
- **Context Recall** — retrieved chunks cover the ground-truth answer
- **Context Precision** — retrieved chunks are on-topic (no noise)

---

## License

MIT
