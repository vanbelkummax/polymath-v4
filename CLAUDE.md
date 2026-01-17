# Polymath v4 - Claude Code Guide

## Current State: POST-REVIEW, PRE-FIX

**Repository:** https://github.com/vanbelkummax/polymath-v4
**Status:** 🔴 Code will not run - 10 critical bugs identified
**Review Date:** 2026-01-17

Three senior engineers audited this codebase and found **29 issues**. The fix plan is in `FIX_ALL_ISSUES.md`.

---

## Quick Reference

| Store | Location | Status |
|-------|----------|--------|
| **Code** | `/home/user/polymath-v4/` | Active development |
| **GitHub** | `vanbelkummax/polymath-v4` | Public repo |
| **Parent System** | `/home/user/polymath-repo/` | Production v3 (reference) |

---

## Critical Issues Summary

| # | Issue | File | Line |
|---|-------|------|------|
| 1 | `Embedder` vs `BGEEmbedder` class name | `lib/embeddings/bge_m3.py` | class def |
| 2 | psycopg2 vs psycopg3 conflict | `lib/db/postgres.py` | all |
| 3 | Missing `PG_POOL_MIN/MAX` config | `lib/config.py` | missing |
| 4 | `detect_all()` API mismatch | `lib/unified_ingest.py` | 259 |
| 5 | Schema column mismatches | `schema/*.sql` vs `lib/*.py` | multiple |
| 6 | Thread-unsafe batch ingestion | `lib/unified_ingest.py` | 377-395 |

**Fix order:** 1 → 2 → 3 → 4 → 5 → 6 (each depends on previous)

---

## Directory Structure

```
polymath-v4/
├── CLAUDE.md                 # THIS FILE - Memory for Claude
├── FIX_ALL_ISSUES.md         # Complete fix prompt (paste into new session)
├── ARCHITECTURE.md           # System design (needs updating after fixes)
├── QUICKSTART.md             # Setup guide
│
├── lib/
│   ├── config.py             # ⚠️ Missing PG_POOL_MIN/MAX
│   ├── unified_ingest.py     # ⚠️ Column names, API mismatch, thread safety
│   ├── db/
│   │   └── postgres.py       # ⚠️ Uses psycopg3, should use psycopg2
│   ├── embeddings/
│   │   └── bge_m3.py         # ⚠️ Class named Embedder, imported as BGEEmbedder
│   ├── ingest/
│   │   ├── pdf_parser.py     # OK (minor: says v3)
│   │   ├── chunking.py       # OK
│   │   └── asset_detector.py # ⚠️ detect_all() expects List[Dict]
│   └── search/
│       └── hybrid_search.py  # ⚠️ Wrong import name
│
├── scripts/
│   ├── ingest_pdf.py         # ⚠️ Duplicates unified_ingest.py logic
│   ├── batch_concepts.py     # ⚠️ Silent error handling
│   ├── extract_skills.py     # ⚠️ Wrong column names
│   ├── github_ingest.py      # ⚠️ Wrong column names, hardcoded paths
│   ├── discover_assets.py    # OK
│   ├── promote_skill.py      # OK
│   ├── sync_neo4j.py         # OK (uses lib/db correctly)
│   └── system_report.py      # OK
│
├── schema/
│   ├── 001_core.sql          # ⚠️ Missing: file_path, content_hash, chunk_index, header
│   ├── 002_concepts.sql      # OK
│   ├── 003_code.sql          # ⚠️ model_id_raw vs model_id, queue_id vs repo_id
│   ├── 004_skills.sql        # ⚠️ Column names don't match extract_skills.py
│   ├── 005_neo4j.cypher      # OK
│   └── 006_advanced.sql      # ⚠️ View references non-existent columns
│
├── skills/                   # Promoted skills
├── skills_drafts/            # Draft skills
└── tests/                    # ❌ MISSING - needs to be created
```

---

## Schema Quick Reference

### documents (001_core.sql)
```sql
doc_id UUID PK, title TEXT, authors TEXT[], year INT,
doi TEXT, pmid TEXT, arxiv_id TEXT, title_hash TEXT,
-- MISSING (need to add):
file_path TEXT, content_hash TEXT, ingested_at TIMESTAMPTZ
```

### passages (001_core.sql)
```sql
passage_id UUID PK, doc_id UUID FK, passage_text TEXT,
page_num INT, embedding vector(1024), is_superseded BOOLEAN
-- MISSING (need to add):
chunk_index INT, header TEXT, char_start INT, char_end INT
```

### repo_queue (003_code.sql)
```sql
queue_id UUID PK,  -- ⚠️ Code uses "repo_id"
repo_url TEXT UNIQUE, owner TEXT, repo_name TEXT,
status TEXT, priority INT, first_seen_doc_id UUID
-- ⚠️ Code expects: repo_id, source_doc_id, source_passage_id
```

### hf_model_mentions (003_code.sql)
```sql
mention_id UUID PK, model_id_raw TEXT,  -- ⚠️ Code uses "model_id"
doc_id UUID FK, passage_id UUID FK
```

### paper_skills (004_skills.sql)
```sql
skill_id UUID PK, skill_name TEXT, description TEXT, steps JSONB,
domain TEXT, status TEXT, source_doc_id UUID, passage_id UUID
-- ⚠️ extract_skills.py uses: skill_description, skill_steps, confidence
```

---

## AI Models Used

| Stage | Model | Cost | Status |
|-------|-------|------|--------|
| Embeddings | BGE-M3 (local GPU) | $0 | ⚠️ Import broken |
| Concepts | Gemini 2.5 Flash Lite (batch) | ~$0.0001/passage | OK |
| Skills | Gemini 2.5 Flash (realtime) | ~$0.001/skill | ⚠️ Column names wrong |
| Reranking | BGE-reranker-v2-m3 (local) | $0 | OK |

---

## Fix Verification Commands

```bash
# After fixes, verify with:
cd /home/user/polymath-v4

# 1. Import tests
python -c "from lib.config import config; print('✓ Config')"
python -c "from lib.embeddings.bge_m3 import BGEEmbedder; print('✓ Embedder')"
python -c "from lib.db.postgres import get_connection; print('✓ DB')"
python -c "from lib.unified_ingest import UnifiedIngestor; print('✓ Ingest')"
python -c "from lib.search.hybrid_search import HybridSearcher; print('✓ Search')"

# 2. Schema tests
for f in schema/*.sql; do psql -U polymath -d polymath -f "$f" && echo "✓ $f"; done

# 3. End-to-end test
python scripts/ingest_pdf.py /path/to/test.pdf
```

---

## Where We've Been

1. **Polymath v1-v3** (`/home/user/polymath-repo/`): Production system with 750K passages, working but complex
2. **v4 Goal**: Lean, execution-ready rewrite with clear architecture
3. **v4 Created**: 2026-01-17, comprehensive but with integration bugs
4. **Review**: 3 senior engineers found 29 issues
5. **Current**: Ready for systematic fixes

## Where We're Going

1. **Immediate**: Fix all 10 critical issues so code runs
2. **Short-term**: Fix high/medium issues, add tests
3. **Medium-term**: Math/algorithm enhancements (Nougat, equation store)
4. **Long-term**: Production deployment, migrate from v3

---

## Key Commands

```bash
# Development
make setup          # Install deps, create .env
make db-init        # Run all schema migrations
make test           # Run verification tests
make health         # System health check

# Ingestion (after fixes)
make ingest PDF=paper.pdf
make concepts       # Submit batch job
make github-queue   # Process GitHub repos

# Git
git add -A && git commit -m "fix: description"
git push origin master
```

---

## Reference: Production v3

If you need to reference working code:
```bash
# v3 location
/home/user/polymath-repo/

# Key working files
/home/user/polymath-repo/lib/hybrid_search_v2.py  # Working search
/home/user/polymath-repo/lib/unified_ingest.py    # Working ingestion
/home/user/polymath-repo/lib/config.py            # Working config
```

---

## Environment

```bash
# Required in .env
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GCP_PROJECT=your-gcp-project-id

# Optional
GITHUB_TOKEN=ghp_xxx
HF_TOKEN=hf_xxx
PG_POOL_MIN=2
PG_POOL_MAX=10
```

---

## Session Start Checklist

When starting a new session on this repo:

1. ☐ Read `FIX_ALL_ISSUES.md` for current fix status
2. ☐ Check which critical issues remain
3. ☐ Run import tests to see current state
4. ☐ Fix issues in order (1→2→3→...)
5. ☐ Commit after each category
6. ☐ Push to GitHub
7. ☐ Update this file with progress
