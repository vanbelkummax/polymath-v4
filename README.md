# Polymath v4

> **An evolving applied knowledge-skills hub for spatial multimodal data analysis—bridging theory, code, and actionable skills across domains.**

## Vision

Polymath is a personal polymathic system. Applications are still being discovered, but the core purpose is clear: **actionable, implementable knowledge that bridges theory and practice.**

This isn't just a paper database. It's:
- A **concept graph** connecting methods across fields (biology ↔ physics ↔ ML)
- A **code-paper bridge** linking implementations to theory
- A **skill repository** capturing successful workflows for reuse
- A **learning accelerator** for conceptual mastery

---

## Current State (2026-01-19)

| Component | Count |
|-----------|-------|
| **Documents** | 2,193 |
| **Passages** | 174,321 (100% embedded) |
| **Repositories** | 1,881 |
| **Concepts** | 2.5M |
| **Neo4j Nodes** | 930K |
| **Neo4j Edges** | 2.7M |

---

## Quick Start

```bash
cd /home/user/polymath-v4

# Search
python scripts/q.py "spatial transcriptomics"
python scripts/q.py "attention mechanisms" --repos

# Ingest
python scripts/ingest_pdf.py paper.pdf
python scripts/ingest_repos.py --source curated

# Status
python scripts/system_report.py --quick
```

---

## Architecture

```
polymath-v4/
├── lib/
│   ├── config.py              # Central config
│   ├── db/postgres.py         # Connection pool
│   ├── embeddings/bge_m3.py   # BGE-M3 embeddings
│   ├── search/hybrid_search.py # Vector + BM25 + reranking
│   └── ingest/                # PDF parsing, chunking, asset detection
├── scripts/                   # CLI tools
├── schema/                    # PostgreSQL migrations
├── skills/                    # Operational skills
└── dashboard/                 # Streamlit UI
```

---

## Key Features

### Hybrid Search
Vector similarity + BM25 keyword matching + optional reranking.

```python
from lib.search.hybrid_search import search
results = search("gene expression prediction", n=10)
```

### Code-Paper Bridge
Find implementations for papers, or papers for code.

```bash
python scripts/q.py "transformer" --code  # Find code for papers
```

### Concept Extraction
Automatic extraction of METHOD, PROBLEM, DOMAIN, ENTITY concepts via Gemini batch API.

```bash
python scripts/batch_concepts.py --submit --limit 100
python scripts/batch_concepts.py --process
```

### Neo4j Graph
Papers → Passages → Concepts with MENTIONS edges for graph traversal.

```bash
python scripts/sync_neo4j.py --full
```

---

## Databases

| Store | Purpose | Connection |
|-------|---------|------------|
| **PostgreSQL** | Documents, passages, embeddings, concepts | `psql -U polymath -d polymath` |
| **Neo4j** | Concept graph | `bolt://localhost:7687` |

---

## Roadmap

- [ ] Test Neo4j graph queries end-to-end
- [ ] Add concepts to GitHub repos (not just papers)
- [ ] Build SIMILAR_TO edges for concept clustering
- [ ] Flashcard generation for learning
- [ ] Gap analysis across polymathic connections

---

## License

MIT

---

## Acknowledgments

Built with [BGE-M3](https://huggingface.co/BAAI/bge-m3), [PostgreSQL](https://postgresql.org) + [pgvector](https://github.com/pgvector/pgvector), [Neo4j](https://neo4j.com), [PyMuPDF](https://pymupdf.readthedocs.io/).
