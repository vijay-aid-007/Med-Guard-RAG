# MedGuard RAG — Production-Grade Medical Question Answering System

> A fully offline, privacy-first Retrieval-Augmented Generation pipeline for medical Q&A with multi-layer guardrails, PII scrubbing, and RAGAS evaluation.

---

## Results at a Glance
|--------------------------------|-----------------------------------|
|            Metric              |              Score                |
|--------------------------------|-----------------------------------|
| Benchmark Accuracy             | **100%** (5/5 clinical questions) |
| RAGAS Faithfulness             |           **0.751**               |
| RAGAS Answer Relevancy         |           **0.675**               |
| RAGAS Context Precision        |         **1.000** (perfect)       |
| RAGAS Context Recall           |           **0.625**               |
| Corpus Size                    |         **40,199 vectors**        |
| Test Suite                     |         **25/25 passing**         |
|--------------------------------|-----------------------------------|

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1 — Input Guardrails                         │
│  PII Scrubber → Jailbreak Check → Domain Filter     │
│  (YAML-driven, 300+ medical anchors, regex patterns)│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2 — Retrieval                                │
│  HyDE Generation → Query Expansion (3 variants)     │
│  → FAISS Search (40,199 vectors, IndexFlatIP)       │
│  → MedCPT Cross-Encoder Reranking                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3 — Generation                               │
│  Prompt Builder → Groq LLM (llama3/gpt-oss)         │
│  → Ollama fallback (phi3:mini, offline)             │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Phase 4 — Output Guardrails                        │
│  Toxicity Check → Overconfidence Detection          │
│  → Grounding Score → PII Scan                       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Phase 5 — Human Handoff                            │
│  Satisfaction Tracker → Slack + Email + PostgreSQL  │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final Answer
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Embedding Model** | `NeuML/pubmedbert-base-embeddings` (768-dim, domain-specific) |
| **Vector Store** | FAISS `IndexFlatIP` (cosine similarity via L2-normalized vectors) |
| **Reranker** | `ncbi/MedCPT-Cross-Encoder` (biomedical cross-encoder) |
| **LLM (primary)** | Groq API — `llama3-8b-8192` / `gpt-oss-120b` |
| **LLM (fallback)** | Ollama — `phi3:mini` (fully offline) |
| **PII Detection** | Microsoft Presidio + custom Indian recognizers (PAN, Aadhaar, Phone) |
| **Framework** | FastAPI + Uvicorn |
| **Session Store** | Redis |
| **Audit DB** | PostgreSQL |
| **Monitoring** | Prometheus + Grafana |
| **Containerization** | Docker + Docker Compose |

---

## Corpus

| Source | Documents | Description |
|---|---|---|
| PubMedQA | 1,043 | Real PubMed abstracts with yes/no/maybe labels |
| MedMCQA | 17,550 | Indian medical entrance exam MCQs with explanations |
| MedQA (USMLE) | 10,004 | USMLE-style clinical reasoning questions |
| PubMed Abstracts | 10,001 | General medical research abstracts |
| PubMed Mechanism | 1,601 | Mechanism-rich abstracts (AMPK, adipokines, pharmacokinetics) |
| **Total** | **40,199 vectors** | 768-dimensional PubMedBERT embeddings |

---

## Key Features

### Multi-Layer Guardrail Pipeline
- **Input guardrails** — Jailbreak detection (7 attack categories), harmful content blocking, semantic domain filtering with 300+ YAML-driven medical anchors
- **Output guardrails** — Toxicity detection, overconfident diagnosis blocking, grounding score validation, PII output scanning
- **Human handoff** — Frustration tracking, Slack webhook, email (SMTP), PostgreSQL audit trail

### Advanced Retrieval
- **HyDE (Hypothetical Document Embedding)** — generates a hypothetical answer before retrieval to bridge question-answer semantic gap
- **Query Expansion** — generates 3 query variants, merges and deduplicates results for broader coverage
- **Cross-encoder Reranking** — MedCPT reranker trained on PubMed data for domain-aligned relevance scoring

### Production-Grade Design
- **Incremental indexing** — add new corpus without full rebuild (`incremental_indexer.py`)
- **YAML-driven patterns** — update guardrail patterns without code changes or redeploy
- **Singleton model loading** — one embedding model instance shared across all components (saves ~300MB RAM)
- **Groq → Ollama fallback** — cloud-fast responses with offline fallback
- **Full observability** — Prometheus metrics, Grafana dashboard, structured logging

### Privacy-First
- Indian PII recognition (PAN card, Aadhaar, mobile numbers) via custom Presidio recognizers
- Both input query and LLM output scanned for PII
- Redaction with labeled placeholders `[NAME_REDACTED]`, `[PAN_REDACTED]`, etc.

---

## Project Structure

```
medguard-rag/
├── data/
│   ├── domain_anchors.yaml     # Medical domain anchors (300+ questions)
│   ├── guardrail_patterns.yaml # Jailbreak + harmful patterns
│   ├── handoff_patterns.yaml   # Satisfaction tracker patterns
│   ├── output_guard_patterns.yaml
│   ├── faiss_index/            # FAISS index + metadata pickle
│   ├── raw/                    # Downloaded datasets (JSONL)
│   └── processed/              # Chunks, eval sets, benchmark results
├── monitoring/
│   ├── prometheus.yml
│   └── grafana_dashboard.json
├── scripts/
│   └── init_db.sql             # PostgreSQL schema (6 tables, 3 views)
├── src/
│   ├── api/                    # FastAPI app, routes, metrics, schemas
│   ├── core/                   # Config, pipeline, session, logging
│   ├── evaluation/             # Benchmark + custom RAGAS evaluator
│   ├── generation/             # LLM client (Groq + Ollama), prompt builder
│   ├── guardrails/             # PII, input, output, human handoff
│   ├── ingestion/              # Loader, chunker, embedder, indexer
│   └── retrieval/              # Retriever (HyDE + expansion), reranker
├── tests/
│   ├── test_e2e.py             # End-to-end pipeline tests
│   ├── test_guardrails.py      # PII, input guard, satisfaction tracker
│   └── test_retrieval.py       # Retriever + reranker tests
├── Dockerfile                  # Multi-stage build, non-root user
├── docker-compose.yml          # API + PostgreSQL + Redis + Ollama + Grafana
└── requirements.txt
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) (for local LLM fallback)
- [Groq API key](https://console.groq.com) (free tier, for primary LLM)

### 1. Clone and Install

```bash
git clone https://github.com/vijay-aid-007/medguard-rag
cd medguard-rag
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
# source .venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set:
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
LLM_MODEL=llama3-8b-8192
```

### 3. Build Knowledge Base

```bash
# Download datasets and build FAISS index (~4 hours on CPU)
python -m src.ingestion.loader
python -m src.ingestion.chunker
python -m src.ingestion.indexer
```

### 4. Start API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the first-line treatment for type 2 diabetes?"}'
```

---

## Docker Deployment

```bash
# Start all services
docker-compose up -d

# Services:
# API        → http://localhost:8000
# Grafana    → http://localhost:3000  (admin/medguard123)
# Prometheus → http://localhost:9090
```

---

## Evaluation

### Benchmark (100% Accuracy)

```bash
python -m src.evaluation.benchmark
```

```
[OK] What is the first-line treatment for type 2 diabetes?
[OK] What is the most common cause of community-acquired pneumonia?
[OK] Which electrolyte abnormality is associated with ACE inhibitors?
[OK] Which vitamin deficiency causes megaloblastic anemia?
[OK] What is the primary mechanism of beta-blockers in hypertension?

Accuracy: 100% (5/5)
```

### RAGAS Evaluation

```bash
python -m src.evaluation.ragas_eval
```

```
Faithfulness      : 0.751
Answer Relevancy  : 0.675
Context Precision : 1.000
Context Recall    : 0.625
```

### Test Suite

```bash
pytest tests/ -v  # 25/25 passing
```

---

## Adding New Corpus (Incremental — No Full Rebuild)

```bash
# 1. Prepare new chunks
python -m src.ingestion.loader          # generates new JSONL
python -m src.ingestion.chunker         # chunks it

# 2. Add to existing index (no rebuild!)
python -m src.ingestion.incremental_indexer \
    --source data/processed/new_chunks.jsonl

# 3. Verify
python -c "
import faiss
idx = faiss.read_index('./data/faiss_index/medguard.index')
print('Total vectors:', idx.ntotal)
"
```

---

## Guardrail Configuration (No Code Changes Needed)

All guardrail patterns are YAML-driven:

```
data/domain_anchors.yaml        # Add new medical specialties
data/guardrail_patterns.yaml    # Add jailbreak/harmful patterns
data/output_guard_patterns.yaml # Add toxic/overconfident patterns
data/handoff_patterns.yaml      # Add frustration/handoff triggers
```

Edit YAML → restart API → changes live. No redeploy needed.

---

## Limitations & Future Work

| Limitation | Future Fix |
|---|---|
| CPU-only embedding (~4hr index build) | GPU acceleration / pre-built index |
| 40k vector corpus | Add more PubMed abstracts incrementally |
| English only | Add multilingual embeddings |
| No conversation memory across sessions | Add Redis conversation history |
| Groq rate limits on free tier | Add request queuing |

---

## Skills Demonstrated

- **RAG Architecture** — HyDE, query expansion, cross-encoder reranking, FAISS
- **Production ML** — singleton patterns, incremental indexing, fallback chains
- **LLM Engineering** — prompt engineering, Groq API, Ollama integration
- **MLOps** — RAGAS evaluation, benchmark harness, Prometheus/Grafana monitoring
- **Security** — PII scrubbing, jailbreak detection, output grounding validation
- **DevOps** — Docker multi-stage build, docker-compose, PostgreSQL, Redis
- **Software Engineering** — YAML-driven config, singleton pattern, factory pattern, pytest

---

## Author

**Vijay** — AI/ML Engineer (2026 Batch, B.Tech CSE + PGP Data Science)

- GitHub: [@vijay-aid-007](https://github.com/vijay-aid-007)
- HuggingFace: [@vijay036](https://huggingface.co/vijay036)