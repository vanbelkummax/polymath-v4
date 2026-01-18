# Polymath v4 - Claude Code Guide

## Current State: OPERATIONAL

**Repository:** https://github.com/vanbelkummax/polymath-v4
**Status:** 🟢 Core pipeline working (ingestion, search, concepts)
**Last Updated:** 2026-01-17

---

## Quick Start

```bash
cd /home/user/polymath-v4

# 1. Ingest a PDF
python scripts/ingest_pdf.py /path/to/paper.pdf

# 2. Search
python -c "from lib.search.hybrid_search import search; print(search('your query', n=5))"

# 3. Extract concepts (batch)
python scripts/batch_concepts.py --submit --limit 100
python scripts/batch_concepts.py --status
```

---

## Skills (Use These!)

Located in `skills/` - load before starting tasks:

| Skill | When to Use |
|-------|-------------|
| `polymath-pdf-ingestion` | Ingesting PDFs (single or batch) |
| `polymath-batch-concepts` | Extracting concepts via Gemini batch API |
| `polymath-search` | Searching with hybrid search, warmup |
| `polymath-smoke-test` | Quick E2E verification |
| `polymath-system-analysis` | Full system health check |

**Example:** Before ingesting papers, read `skills/polymath-pdf-ingestion.md`

---

## Architecture

```
polymath-v4/
├── lib/
│   ├── config.py              # Central config, loads .env
│   ├── embeddings/bge_m3.py   # BGE-M3 embeddings (thread-safe)
│   ├── search/hybrid_search.py # Vector + BM25 + reranking
│   ├── ingest/
│   │   ├── pdf_parser.py      # PyMuPDF text extraction
│   │   ├── chunking.py        # Text chunking
│   │   └── asset_detector.py  # GitHub/HF/citation detection
│   └── db/postgres.py         # Database connections
├── scripts/
│   ├── ingest_pdf.py          # PDF ingestion CLI
│   ├── batch_concepts.py      # Gemini batch concept extraction
│   ├── system_report.py       # Health check
│   └── ...
├── schema/                    # PostgreSQL migrations
├── skills/                    # Operational skills (READ THESE)
└── tests/                     # Test suite
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill in:

```bash
# Required
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_password_here

# Google Cloud (for batch concepts)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT=your-gcp-project-id
GCP_LOCATION=us-central1
GCS_BUCKET=your-bucket-name

# Optional
GITHUB_TOKEN=ghp_xxx
HF_TOKEN=hf_xxx
```

---

## Key Commands

### Ingestion
```bash
# Single PDF
python scripts/ingest_pdf.py paper.pdf

# Batch with parallel workers
python scripts/ingest_pdf.py /path/to/*.pdf --workers 2

# Skip embeddings (faster, for testing)
python scripts/ingest_pdf.py paper.pdf --no-embeddings
```

### Search
```python
from lib.search.hybrid_search import warmup, search

# Fast repeated queries (warmup once)
searcher = warmup(rerank=True)
results = searcher.hybrid_search("spatial transcriptomics", n=10)

# Quick one-off search
results = search("gene expression prediction", n=5)
```

### Concept Extraction
```bash
# Check pending passages
python scripts/batch_concepts.py

# Submit batch job (50-500 passages recommended)
python scripts/batch_concepts.py --submit --limit 100

# Check job status
python scripts/batch_concepts.py --status

# Process completed jobs
python scripts/batch_concepts.py --process
```

### System Health
```bash
python scripts/system_report.py --quick
```

---

## Database

### Quick Counts
```sql
SELECT
    (SELECT COUNT(*) FROM documents) as docs,
    (SELECT COUNT(*) FROM passages) as passages,
    (SELECT COUNT(embedding) FROM passages) as embedded,
    (SELECT COUNT(*) FROM passage_concepts) as concepts;
```

### Recent Ingestion
```sql
SELECT title, created_at
FROM documents
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;
```

### Core Schema
```sql
documents (doc_id, title, authors, year, doi, pmid, title_hash)
passages (passage_id, doc_id, passage_text, embedding, is_superseded)
passage_concepts (passage_id, concept_name, concept_type, confidence)
```

---

## AI Models

| Component | Model | Notes |
|-----------|-------|-------|
| Embeddings | BGE-M3 (local GPU) | 1024-dim, thread-safe |
| Concepts | gemini-2.5-flash-lite | Batch API, 50% cheaper |
| Reranking | bge-reranker-v2-m3 | Optional, improves relevance |

---

## Performance Notes

### Search Latency
- **First query:** ~100s (model loading)
- **With warmup:** ~6s warmup, then ~7s per query
- **Without reranking:** ~2s per query

### Ingestion Throughput
- **Single PDF:** ~10s (with embeddings)
- **Batch (2 workers):** ~8s per PDF
- **GPU bound:** RTX 5090 handles ~100 passages/sec

---

## Troubleshooting

### "Cannot copy out of meta tensor"
Fixed in current version. Use `--workers 1` if issue persists.

### Batch job fails
Check JSONL format - must have `"request"` wrapper. Current version is fixed.

### Search returns nothing
```bash
# Check embeddings exist
psql -U polymath -d polymath -c "SELECT COUNT(embedding) FROM passages;"
```

### Slow first query
Use `warmup()` at application start:
```python
from lib.search.hybrid_search import warmup
searcher = warmup()  # Do this once at startup
```

---

## Development

### Run Tests
```bash
python -c "from lib.config import config; print('✓ Config')"
python -c "from lib.embeddings.bge_m3 import BGEEmbedder; print('✓ Embedder')"
python -c "from lib.search.hybrid_search import HybridSearcher; print('✓ Search')"
```

### Commit Convention
```bash
git add -A
git commit -m "fix: description of fix"
git push origin master
```

---

## Session Checklist

1. ☐ Check batch job status: `python scripts/batch_concepts.py --status`
2. ☐ Run smoke test: Read `skills/polymath-smoke-test.md`
3. ☐ Review pending tasks in this file
4. ☐ Commit and push changes

---

## Pending Tasks

| Task | Priority | Notes |
|------|----------|-------|
| Process batch concept results | High | When job completes |
| Add --process implementation | High | Parse Gemini batch output |
| Fix remaining FIX_ALL_ISSUES.md items | Medium | Schema alignment |
| Add test suite | Medium | Basic import/integration tests |
| GitHub ingestion | Low | Queue processing |

---

## Related Resources

| Resource | Location |
|----------|----------|
| Skills | `skills/` directory |
| Test Plan | `TEST_PLAN.md` |
| Architecture | `ARCHITECTURE.md` |
| Fix List | `FIX_ALL_ISSUES.md` |
