# Polymath v4 - Claude Code Guide

## Current State: OPERATIONAL

**Status:** 🟢 Production-ready (ingestion, search, concepts, repos)
**Last Updated:** 2026-01-18

---

## Quick Start

```bash
cd /home/user/polymath-v4

# 1. Ingest a PDF
python scripts/ingest_pdf.py /path/to/paper.pdf

# 2. Search
python -c "from lib.search.hybrid_search import search; print(search('your query', n=5))"

# 3. System health
python scripts/system_report.py --quick
```

---

## Architecture

```
polymath-v4/
├── lib/
│   ├── config.py              # Central config (thread-safe)
│   ├── db/postgres.py         # Connection pool (thread-safe)
│   ├── embeddings/bge_m3.py   # BGE-M3 embeddings (thread-safe)
│   ├── search/hybrid_search.py # Vector + BM25 + reranking
│   └── ingest/
│       ├── pdf_parser.py      # PyMuPDF text extraction
│       ├── chunking.py        # Header-aware chunking
│       └── asset_detector.py  # GitHub/HF/citation detection
├── scripts/                   # CLI tools
├── schema/                    # PostgreSQL migrations (001-008)
├── dashboard/                 # Streamlit UI
└── tests/                     # 26 tests
```

---

## Configuration

### Required Environment Variables

```bash
# .env file
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=polymathic2026

# Google Cloud (for batch concepts)
GOOGLE_APPLICATION_CREDENTIALS=/home/user/.gcp/service-account.json
GCP_PROJECT=fifth-branch-483806-m1
GCS_BUCKET=polymath-batch-jobs
```

### Search Tuning (Optional)

```bash
# These can be set in .env or as environment variables
SEARCH_VECTOR_WEIGHT=0.7        # 0-1, higher = more semantic
SEARCH_CANDIDATE_MULTIPLIER=3   # Candidates = n * multiplier
SEARCH_RRF_K=60                 # RRF fusion constant
SEARCH_GRAPHRAG_MAX_EXPANSIONS=5
SEARCH_GRAPHRAG_MIN_COOCCURRENCE=3
```

---

## Key Commands

### Ingestion
```bash
# Single PDF
python scripts/ingest_pdf.py paper.pdf

# Batch with parallel workers
python scripts/ingest_pdf.py /path/to/*.pdf --workers 4

# With Zotero metadata
python scripts/ingest_pdf.py paper.pdf --zotero-csv metadata.csv
```

### Search
```python
from lib.search.hybrid_search import warmup, search

# Fast repeated queries (warmup once at startup)
searcher = warmup(rerank=True)
results = searcher.hybrid_search("spatial transcriptomics", n=10)

# With GraphRAG expansion
results = searcher.hybrid_search("gene expression", graph_expand=True)

# Override default weights
results = searcher.hybrid_search("query", vector_weight=0.8)

# Quick one-off
results = search("gene expression prediction", n=5)
```

### Concepts
```bash
python scripts/batch_concepts.py --submit --limit 100
python scripts/batch_concepts.py --status
python scripts/batch_concepts.py --process
```

### Discovery
```bash
# Find papers via CORE API
python scripts/discover_papers.py "spatial transcriptomics" --auto-ingest

# Gap analysis
python scripts/active_librarian.py --analyze-gaps
```

---

## Database

### Quick Stats
```sql
SELECT
    (SELECT COUNT(*) FROM documents) as docs,
    (SELECT COUNT(*) FROM passages) as passages,
    (SELECT COUNT(embedding) FROM passages) as embedded,
    (SELECT COUNT(*) FROM passage_concepts) as concepts,
    (SELECT COUNT(*) FROM repositories) as repos;
```

### Schema (key tables)
```sql
documents (doc_id, title, authors, year, doi, pmid, title_hash)
passages (passage_id, doc_id, passage_text, embedding, is_superseded)
passage_concepts (passage_id, concept_name, concept_type, confidence)
repositories (repo_id, url, name, stars, language)
repo_passages (passage_id, repo_id, passage_text, embedding)
```

### Performance Indexes
```sql
-- Partial indexes for active passages (schema/008)
idx_passages_active              -- Non-superseded passages
idx_passages_active_embedding    -- Vector search on active only
idx_documents_created_at         -- Recent docs
idx_pc_high_confidence           -- High-confidence concepts
```

---

## Thread Safety

All core modules are thread-safe:

| Module | Mechanism |
|--------|-----------|
| `db/postgres.py` | Connection pool with `_pool_lock` |
| `embeddings/bge_m3.py` | `_model_lock` + `_encode_lock` |
| `search/hybrid_search.py` | Uses pooled connections |

---

## AI Models

| Component | Model | Notes |
|-----------|-------|-------|
| Embeddings | BGE-M3 | 1024-dim, local GPU |
| Concepts | gemini-2.5-flash-lite | Batch API (50% cheaper) |
| Reranking | bge-reranker-v2-m3 | Optional cross-encoder |

---

## Performance

### Search Latency
- **Cold start:** ~100s (model loading)
- **With warmup:** ~6s warmup, ~7s/query
- **Without reranking:** ~2s/query

### Ingestion
- **Single PDF:** ~10s (with embeddings)
- **Batch (4 workers):** ~3s per PDF
- **GPU:** RTX 5090, ~100 passages/sec

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Quick import check
python -c "from lib.config import config; print('✓')"
python -c "from lib.db.postgres import check_health; print(check_health())"
python -c "from lib.search.hybrid_search import HybridSearcher; print('✓')"
```

---

## Troubleshooting

### Search returns nothing
```bash
psql -U polymath -d polymath -c "SELECT COUNT(embedding) FROM passages;"
```

### Slow first query
```python
from lib.search.hybrid_search import warmup
searcher = warmup()  # Call once at startup
```

### Connection issues
```python
from lib.db.postgres import check_health
print(check_health())  # Should show status: healthy
```

---

## Migrations

Run in order if setting up fresh:
```bash
psql -U polymath -d polymath -f schema/001_core.sql
psql -U polymath -d polymath -f schema/002_concepts.sql
psql -U polymath -d polymath -f schema/003_code.sql
psql -U polymath -d polymath -f schema/006_advanced.sql
psql -U polymath -d polymath -f schema/007_repositories.sql
psql -U polymath -d polymath -f schema/008_performance_indexes.sql
```

---

## Current Stats (2026-01-18)

| Metric | Count |
|--------|-------|
| Documents | 1,778 |
| Passages | 148,208 |
| Concepts | 4,829,145 (batch job running for new passages) |
| Repositories | 1,800 |

---

## Related Files

| Resource | Location |
|----------|----------|
| Main config | `lib/config.py` |
| Search tuning | Environment variables |
| Skills | `skills/` directory |
| Dashboard | `streamlit run dashboard/app.py` |
