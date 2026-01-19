# Polymath v4 - Claude Code Guide

## Current State: PRODUCTION READY

**Status:** 🟢 Verified production-ready (2026-01-19 audit)
**Last Audit:** 2026-01-19
**Auditor:** Claude Opus 4.5

---

## Quick Start

```bash
cd /home/user/polymath-v4

# 1. Search (papers)
python scripts/q.py "spatial transcriptomics"

# 2. Search (code-paper bridge)
python scripts/q.py "gene expression prediction" --code

# 3. Search (repos)
python scripts/q.py "graph neural network" --repos

# 4. System health
python scripts/system_report.py --quick
```

---

## Current Statistics (2026-01-19)

| Metric | Count | Status |
|--------|-------|--------|
| Documents | 2,193 | ✅ |
| Passages | 174,321 | ✅ 100% embedded |
| Concepts | 7,362,693 | ✅ |
| Repositories | 1,881 | ✅ |
| Code Chunks | 578,830 | ✅ |
| Paper-Repo Links | 524 | ✅ |
| Neo4j Nodes | 1.1M+ | ✅ |

---

## Architecture

```
polymath-v4/
├── lib/
│   ├── config.py              # Central config (thread-safe) ✅
│   ├── db/postgres.py         # Connection pool (thread-safe) ✅
│   ├── embeddings/bge_m3.py   # BGE-M3 embeddings (thread-safe) ✅
│   ├── search/hybrid_search.py # Vector + BM25 + reranking ✅
│   ├── unified_ingest.py      # Main ingestion orchestrator ✅
│   └── ingest/
│       ├── pdf_parser.py      # PyMuPDF text extraction
│       ├── chunking.py        # Header-aware chunking
│       └── asset_detector.py  # GitHub/HF/citation detection ✅
├── scripts/                   # CLI tools (28 scripts)
├── schema/                    # PostgreSQL migrations (001-009)
├── skills/                    # Operational skills
├── dashboard/                 # Streamlit UI
└── tests/                     # 26 tests
```

---

## Services

| Service | Status | Connection |
|---------|--------|------------|
| PostgreSQL | ✅ Running | `psql -U polymath -d polymath` |
| Neo4j | ✅ Running | `bolt://localhost:7687` |

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

### Connection Pool Settings (lib/config.py)

```python
PG_POOL_MIN = 2   # Minimum connections
PG_POOL_MAX = 10  # Maximum connections
```

### Search Tuning (Optional)

```bash
SEARCH_VECTOR_WEIGHT=0.7        # 0-1, higher = more semantic
SEARCH_CANDIDATE_MULTIPLIER=3   # Candidates = n * multiplier
SEARCH_RRF_K=60                 # RRF fusion constant
SEARCH_GRAPHRAG_MAX_EXPANSIONS=5
SEARCH_GRAPHRAG_MIN_COOCCURRENCE=3
```

---

## Key Commands

### Search

```bash
# Paper search (semantic)
python scripts/q.py "spatial transcriptomics"

# Code-Paper Bridge (find repos for papers)
python scripts/q.py "gene expression from H&E" --code

# Repo search (find code mentioning topic)
python scripts/q.py "transformer" --repos

# Fast mode (skip reranking)
python scripts/q.py "query" --fast
```

### Ingestion

```bash
# Single PDF
python scripts/ingest_pdf.py paper.pdf

# Batch with parallel workers
python scripts/ingest_pdf.py /path/to/*.pdf --workers 4
```

### Concepts (Batch API)

```bash
python scripts/batch_concepts.py --submit --limit 100
python scripts/batch_concepts.py --status
python scripts/batch_concepts.py --process
```

### Neo4j Sync

```bash
python scripts/sync_neo4j.py --full
python scripts/sync_neo4j.py --incremental
```

### System Health

```bash
python scripts/system_report.py --quick
python scripts/system_report.py  # Full report
```

---

## Thread Safety (Verified 2026-01-19)

All core modules are thread-safe:

| Module | Mechanism | Status |
|--------|-----------|--------|
| `lib/db/postgres.py` | `_pool_lock` + double-check locking | ✅ Verified |
| `lib/embeddings/bge_m3.py` | `_model_lock` + `_encode_lock` | ✅ Verified |
| `lib/unified_ingest.py` | Per-task connections | ✅ Verified |
| `lib/search/hybrid_search.py` | Uses pooled connections | ✅ Verified |

---

## Database Schema (Key Tables)

```sql
-- Core
documents (doc_id, title, authors, year, doi, pmid, title_hash, pdf_path)
passages (passage_id, doc_id, passage_text, embedding, section, page_num)
passage_concepts (passage_id, concept_name, concept_type, confidence)

-- Code
repositories (repo_id, repo_url, owner, name, stars, language)
repo_passages (passage_id, repo_id, passage_text, embedding)
code_chunks (chunk_id, file_id, chunk_type, name, content)

-- Links
paper_repo_links (doc_id, repo_id, link_type, confidence)
paper_repos (doc_id, repo_url, detection_method, confidence)

-- Skills
paper_skills (skill_id, skill_name, skill_type, description, evidence_count, status)
```

---

## AI Models

| Component | Model | Location | Notes |
|-----------|-------|----------|-------|
| Embeddings | BGE-M3 | Local GPU | 1024-dim, $0 |
| Concepts | Gemini 2.5 Flash Lite | Batch API | 50% cheaper |
| Reranking | bge-reranker-v2-m3 | Local GPU | Optional |

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
python -c "from lib.config import config; print(f'Pool: {config.PG_POOL_MIN}-{config.PG_POOL_MAX}')"
python -c "from lib.embeddings.bge_m3 import BGEEmbedder, BGEM3Embedder, Embedder; print('Aliases OK')"
python -c "from lib.db.postgres import get_connection, get_pool; print('DB OK')"
python -c "from lib.search.hybrid_search import HybridSearcher; print('Search OK')"
```

---

## Troubleshooting

### Neo4j not running
```bash
docker restart polymax-neo4j
sleep 20  # Wait for startup
```

### Search returns nothing
```bash
psql -U polymath -d polymath -c "SELECT COUNT(embedding) FROM passages;"
```

### Slow first query
```python
from lib.search.hybrid_search import warmup
searcher = warmup()  # Call once at startup
```

### Connection pool exhausted
Check `PG_POOL_MAX` in `lib/config.py` and increase if needed.

---

## Migrations

Run in order if setting up fresh:
```bash
psql -U polymath -d polymath -f schema/001_core.sql
psql -U polymath -d polymath -f schema/002_concepts.sql
psql -U polymath -d polymath -f schema/003_code.sql
psql -U polymath -d polymath -f schema/004_skills.sql
psql -U polymath -d polymath -f schema/006_advanced.sql
psql -U polymath -d polymath -f schema/007_repositories.sql
psql -U polymath -d polymath -f schema/008_performance_indexes.sql
psql -U polymath -d polymath -f schema/009_algorithm_registry.sql
psql -U polymath -d polymath -f schema/010_stabilization_fixes.sql
```

---

## Audit History

| Date | Auditor | Status | Notes |
|------|---------|--------|-------|
| 2026-01-19 | Claude Opus 4.5 | ✅ PASS | Full stabilization audit |

See `docs/audits/STABILIZATION_AUDIT_2026_01_19.md` for details.

---

## Related Files

| Resource | Location |
|----------|----------|
| Main config | `lib/config.py` |
| System architecture | `ARCHITECTURE.md` |
| Stabilization audit | `docs/audits/STABILIZATION_AUDIT_2026_01_19.md` |
| Skills | `skills/` directory |
| Dashboard | `streamlit run dashboard/app.py` |
