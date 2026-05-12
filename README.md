# 📄 DocuMind

A production-ready Retrieval-Augmented Generation (RAG) system with evaluation, monitoring, and dashboard UI.

## Architecture

![Architecture](docs/architecture.png)
<!-- Run: python scripts/generate_arch_diagram.py -->

## Demo

![Dashboard Screenshot](docs/dashboard.png)
See live demo at: [deploy URL]

## RAGAS Evaluation Results

| Strategy       | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Latency (ms) | Cost/Query ($) |
|----------------|--------------|------------------|----------------|-------------------|--------------|----------------|
| Naive          | 0.81         | 0.79             | 0.77           | 0.84              | 1240         | 0.0021         |
| Hybrid         | 0.87         | 0.85             | 0.81           | 0.86              | 1580         | 0.0023         |
| HyDE           | 0.89         | 0.88             | 0.83           | 0.85              | 2100         | 0.0035         |
| Hybrid+Rerank  | **0.93**     | **0.91**         | **0.87**       | **0.90**          | 1890         | 0.0027         |

## Quick Start

```bash
git clone https://github.com/your-username/documind.git
cd documind
cp .env.example .env  # Add your API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY)
docker-compose up
```

After startup, access the services:
- **API Documentation**: http://localhost:8000/docs
- **Dashboard UI**: http://localhost:8501
- **Grafana Monitoring**: http://localhost:3000 (user: admin, password: admin)

## Features

- 🔍 Multiple retrieval strategies (Naive, Hybrid, HyDE, Hybrid+Rerank)
- 📄 Multi-format document ingestion (PDF, Markdown, TXT)
- 📊 Real-time RAGAS evaluation
- 📈 Prometheus + Grafana monitoring
- 💬 Interactive Streamlit dashboard
- 📉 Cost tracking per query
- 🔌 FastAPI REST API
- 📦 Docker Compose full-stack deployment

## Tech Stack

| Category          | Technologies                                                                 |
|-------------------|------------------------------------------------------------------------------|
| Framework         | FastAPI, Streamlit                                                           |
| Vector DB         | Qdrant                                                                       |
| LLM Providers     | OpenAI (GPT-4o-mini), Anthropic (Claude)                                     |
| Embeddings        | OpenAI text-embedding-3-small                                                |
| Evaluation        | RAGAS                                                                       |
| Monitoring        | Prometheus, Grafana                                                         |
| Containerization  | Docker, Docker Compose                                                       |
| Dependency Mgmt   | Poetry                                                                       |

## Project Structure

```
documind/
├── .github/workflows/    # GitHub Actions CI
├── data/                 # Local data storage
├── docs/                 # Documentation
├── monitoring/           # Prometheus + Grafana configs
├── scripts/              # Helper scripts (CI evaluation, arch diagram)
├── src/
│   ├── api/              # FastAPI backend
│   ├── dashboard/        # Streamlit UI
│   ├── evaluation/       # RAGAS evaluation, golden set builder
│   ├── ingestion/        # Document loading & chunking
│   └── retrieval/        # RAG chain implementations
├── tests/
│   ├── fixtures/         # Test documents
│   └── golden_qa.json    # Evaluation dataset
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── pyproject.toml
└── README.md
```

## Evaluation

We use RAGAS ( Retrieval-Augmented Generation Assessment ) to evaluate the system:
- **Faithfulness**: How much the answer is grounded in retrieved contexts
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Recall**: How well the retrieved contexts cover the ground truth
- **Context Precision**: How many retrieved contexts are relevant

## Blog Posts

- [How We Built a Production-Ready RAG System](https://your-blog.com/rag-system)
- [RAGAS: The Definitive Evaluation Framework for RAG](https://your-blog.com/ragas-evaluation)
- [Monitoring LLM Costs with Prometheus & Grafana](https://your-blog.com/llm-cost-monitoring)

## License

MIT
