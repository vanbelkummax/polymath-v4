# Polymath v4: Polymathic Knowledge Base

> **A Postgres-first knowledge system for scientific papers, code repositories, and cross-domain insights.**

<p align="center">
  <img src="docs/images/architecture.png" alt="Polymath v4 Architecture" width="800">
</p>

<p align="center">
  <em>Document → Process → Store → Search: The Polymath pipeline transforms scientific papers into a searchable knowledge graph.</em>
</p>

## Overview

Polymath v4 is a **polymathic knowledge system** designed to:

1. **Ingest scientific papers** with rich metadata from Zotero, PDFs, and external APIs
2. **Extract assets** (GitHub repos, HuggingFace models, DOI citations) from paper text
3. **Enable semantic search** using BGE-M3 embeddings stored in PostgreSQL pgvector
4. **Build cross-domain connections** via concept extraction and Neo4j graph

### Key Statistics (2026-01-18)

| Metric | Count |
|--------|-------|
| Documents | 1,698 |
| Paper Passages | 143,103 |
| Repositories | 1,791 |
| Repo Passages | 51,006 |
| Paper-Repo Links | 423 |

---

## Three-Stage Pipeline

<p align="center">
  <img src="docs/images/pipeline.png" alt="Ingestion Pipeline" width="800">
</p>

---

## Metadata Waterfall

Polymath uses a **waterfall approach** for metadata extraction, preferring richer sources:

<p align="center">
  <img src="docs/images/waterfall.png" alt="Metadata Waterfall" width="800">
</p>

---

## Deduplication Strategy

**Strict approach:** Prefer duplicates over losing unique papers.

<p align="center">
  <img src="docs/images/deduplication.png" alt="Deduplication Checks" width="800">
</p>

---

## Quick Start

### 1. Ingest a Single PDF

```bash
cd /home/user/polymath-v4
python scripts/ingest_pdf.py /path/to/paper.pdf
```

### 2. Batch Ingest with Zotero Metadata

```bash
# Prepare Zotero CSV (deduplicate, map paths)
python scripts/prepare_zotero_ingest.py '/path/to/My Library.csv'

# Ingest PDFs with rich metadata
python scripts/ingest_pdf.py /path/to/pdfs/ \
    --recursive \
    --zotero-csv /path/to/zotero_prepared.csv \
    --workers 2
```

### 3. Search the Knowledge Base

```python
from lib.search.hybrid_search import search

# Semantic search
results = search("spatial transcriptomics methods", n=10)
for r in results:
    print(f"[{r['score']:.3f}] {r['title']}")
```

### 4. Find Papers with Code

```python
from lib.db.postgres import get_connection

conn = get_connection()
cur = conn.cursor()
cur.execute("""
    SELECT d.title, d.year, r.repo_url
    FROM documents d
    JOIN paper_repos r ON d.doc_id = r.doc_id
    WHERE r.verified = true
    ORDER BY d.year DESC
    LIMIT 10
""")
for row in cur.fetchall():
    print(f"{row[0]} ({row[1]}): {row[2]}")
```

---

## Database Schema

<p align="center">
  <img src="docs/images/schema.png" alt="Core Tables" width="800">
</p>

---

## Asset Detection

Polymath automatically detects assets mentioned in papers:

| Asset Type | Detection Pattern | Example |
|------------|-------------------|---------|
| **GitHub Repos** | `github.com/owner/repo` | `github.com/scverse/squidpy` |
| **HuggingFace Models** | `huggingface.co/model` | `facebook/dinov2-large` |
| **DOI Citations** | `10.xxxx/...` | `10.1038/s41586-025-09025-8` |
| **arXiv Papers** | `arxiv.org/abs/xxxx` | `2401.12345` |

---

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Single PDF ingest | ~1.2s | With embeddings |
| Batch ingest (2 workers) | ~0.6s/PDF | Parallel processing |
| Semantic search | ~2s | First query (model loading) |
| Semantic search | ~0.5s | Subsequent queries |

---

## Project Structure

```
polymath-v4/
├── lib/
│   ├── config.py              # Central configuration
│   ├── embeddings/bge_m3.py   # BGE-M3 embeddings (thread-safe)
│   ├── search/hybrid_search.py # Vector + BM25 + reranking
│   ├── ingest/
│   │   ├── pdf_parser.py      # PyMuPDF text extraction
│   │   ├── chunking.py        # Text chunking
│   │   └── asset_detector.py  # GitHub/HF/citation detection
│   └── db/postgres.py         # Database connections
├── scripts/
│   ├── ingest_pdf.py          # PDF ingestion CLI
│   ├── prepare_zotero_ingest.py # Zotero CSV preparation
│   ├── batch_concepts.py      # Gemini batch concept extraction
│   └── system_report.py       # Health check
├── schema/                    # PostgreSQL migrations
├── skills/                    # Operational skills for Claude
└── docs/plans/                # Design documents
```

---

## Environment Variables

```bash
# Required
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql

# Optional - for enrichment
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_password
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## Skills (for Claude Code)

| Skill | Purpose |
|-------|---------|
| `polymath-pdf-ingestion` | Single/batch PDF ingestion |
| `polymath-fresh-ingest` | Full pipeline with Zotero |
| `polymath-search` | Semantic search operations |
| `polymath-batch-concepts` | Gemini concept extraction |
| `polymath-smoke-test` | End-to-end verification |

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with:
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multilingual embeddings
- [PostgreSQL](https://postgresql.org) + [pgvector](https://github.com/pgvector/pgvector) - Vector storage
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF parsing
- [Neo4j](https://neo4j.com) - Graph database for concepts
