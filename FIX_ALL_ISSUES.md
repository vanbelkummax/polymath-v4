# Polymath v4 - Complete Fix Prompt

**Copy everything below this line and paste into a new Claude Code session:**

---

## Context

I have a Polymath v4 repository at `/home/user/polymath-v4/` that was reviewed by 3 senior engineers. They found **29 issues** (10 critical, 7 high, 6 medium, 4 enhancements). The code currently **will not run** due to critical bugs.

**Repository:** https://github.com/vanbelkummax/polymath-v4

## Your Mission

Fix ALL issues in priority order. After each fix, verify it works. Commit after each major category is complete.

---

## CRITICAL FIXES (Do These First - Code Won't Run Without Them)

### 1. Fix Embedder Class Name Mismatch

**Problem:** `lib/embeddings/bge_m3.py` defines class `Embedder`, but 3 files import `BGEEmbedder` or `BGEM3Embedder`.

**Files to fix:**
- `lib/embeddings/bge_m3.py` - Rename class to `BGEEmbedder` and add alias
- `lib/search/hybrid_search.py:15` - imports `BGEEmbedder`
- `scripts/ingest_pdf.py:47` - imports `BGEM3Embedder`
- `lib/unified_ingest.py:29` - imports `BGEEmbedder`

**Also fix method names:** The class has `encode()` but callers use `embed_single()` and `embed_batch()`. Standardize to provide all three methods.

### 2. Fix psycopg2/psycopg3 Conflict

**Problem:** `lib/db/postgres.py` uses psycopg3 (`psycopg`, `psycopg_pool`, `pgvector.psycopg`) but:
- All other files use `psycopg2`
- `requirements.txt` only has `psycopg2-binary`

**Decision:** Convert `lib/db/postgres.py` to use psycopg2 (simpler, matches everything else).

**Files to fix:**
- `lib/db/postgres.py` - Rewrite to use psycopg2 with a simple connection pool
- `requirements.txt` - Keep psycopg2-binary, add pgvector

### 3. Add Missing Config Parameters

**Problem:** `lib/db/postgres.py:43-44` references `config.PG_POOL_MIN` and `config.PG_POOL_MAX` which don't exist.

**File to fix:** `lib/config.py` - Add:
```python
PG_POOL_MIN: int = int(os.environ.get("PG_POOL_MIN", "2"))
PG_POOL_MAX: int = int(os.environ.get("PG_POOL_MAX", "10"))
```

### 4. Fix Asset Detection API Mismatch

**Problem:** `lib/unified_ingest.py:259` calls:
```python
assets = detector.detect_all(chunk['content'])  # Passes string
```
But `AssetDetector.detect_all()` in `lib/ingest/asset_detector.py:101` expects `List[Dict]`.

**Files to fix:**
- `lib/ingest/asset_detector.py` - Add a simpler `detect_from_text(text: str)` method
- `lib/unified_ingest.py:259` - Use the new method

### 5. Fix Schema Column Mismatches (MAJOR)

**Problem:** Multiple files use column names that don't exist in schema.

**Schema file:** `schema/001_core.sql`
- Has: `pdf_path`, `passage_index`
- Missing: `file_path`, `content_hash`, `ingested_at`, `chunk_index`, `header`

**Fix:** Update schema/001_core.sql to add missing columns:
```sql
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_path TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE passages ADD COLUMN IF NOT EXISTS chunk_index INTEGER;
ALTER TABLE passages ADD COLUMN IF NOT EXISTS header TEXT;
```

**Schema file:** `schema/003_code.sql`
- `hf_model_mentions` has `model_id_raw` but `unified_ingest.py:277` inserts `model_id`
- `repo_queue` has `queue_id` but `github_ingest.py` and `006_advanced.sql` use `repo_id`

**Fix:** Align column names - prefer what the code uses, update schema.

### 6. Fix extract_skills.py Column Names

**Problem:** `scripts/extract_skills.py:110-151` inserts:
- `skill_description`, `skill_steps`, `confidence`

But `schema/004_skills.sql` defines:
- `description`, `steps`, `status`

**Fix:** Update extract_skills.py to use correct column names.

### 7. Fix v_repo_paper_links View

**Problem:** `schema/006_advanced.sql:119-124` view references `repo_id` columns that don't exist.

**Fix:** Rewrite view to use correct column names (`queue_id`, etc.).

### 8. Fix Thread Safety in Batch Ingestion

**Problem:** `lib/unified_ingest.py:377-395` shares a single psycopg2 connection across ThreadPoolExecutor workers. psycopg2 connections aren't thread-safe.

**Fix:** Create connection per task OR use a proper connection pool.

---

## HIGH PRIORITY FIXES

### 9. Update requirements.txt

Add missing dependencies:
```
FlagEmbedding>=1.2.0
pgvector>=0.2.0
google-generativeai>=0.3.0
```

### 10. Unify Database Connections

**Problem:** Scripts create raw `psycopg2.connect()` instead of using `lib/db/postgres.py`.

**Files to fix:**
- `scripts/ingest_pdf.py`
- `scripts/github_ingest.py`
- `scripts/batch_concepts.py`
- `scripts/extract_skills.py`
- `lib/unified_ingest.py`

All should use `from lib.db.postgres import get_connection` (or whatever you name it).

### 11. Remove Duplicate Ingestion Logic

**Problem:** `scripts/ingest_pdf.py` duplicates logic from `lib/unified_ingest.py`.

**Fix:** Make `scripts/ingest_pdf.py` a thin wrapper that calls `UnifiedIngestor`.

### 12. Fix Hardcoded Paths

**Problem:** `scripts/github_ingest.py` redefines `REPOS_DIR` with hardcoded path.

**Fix:** Remove redefinition, use `from lib.config import config` everywhere.

### 13. Create Missing Files or Update Docs

**ARCHITECTURE.md references these files that don't exist:**
- `lib/ingest/pipeline.py`
- `lib/ingest/concept_extractor.py`
- `lib/ingest/skill_extractor.py`
- `lib/search/reranker.py`
- `lib/db/neo4j.py`

**Fix:** Either create stub files OR update ARCHITECTURE.md to match reality.

### 14. Add Basic Test Suite

Create `tests/` directory with:
- `test_imports.py` - Can all modules import?
- `test_chunking.py` - Does chunking work?
- `test_embedder.py` - Does BGE-M3 produce 1024-dim vectors?
- `test_schema.py` - Can migrations run?

---

## MEDIUM PRIORITY

### 15. Fix Version Comments
- `lib/ingest/pdf_parser.py:3` says "v3" → change to "v4"
- `lib/embeddings/bge_m3.py:2` says "v3" → change to "v4"

### 16. Improve Error Handling
- `scripts/batch_concepts.py:272-273` uses `logger.debug()` for errors → use `logger.warning()`

### 17. Move Prompts to Separate File
Create `lib/prompts.py` and move `CONCEPT_PROMPT` from `batch_concepts.py` and `SKILL_PROMPT` from `extract_skills.py`.

### 18. Add GIN Index
In `schema/003_code.sql`, add:
```sql
CREATE INDEX IF NOT EXISTS idx_code_chunks_concepts ON code_chunks USING GIN (concepts);
```

### 19. Document PAT_2 Environment Variable
`scripts/github_ingest.py:44` references `PAT_2` - document it or remove it.

---

## VERIFICATION

After all fixes, run:
```bash
# Test imports
python -c "from lib.config import config; print('Config OK')"
python -c "from lib.embeddings.bge_m3 import BGEEmbedder; print('Embedder OK')"
python -c "from lib.db.postgres import get_connection; print('DB OK')"
python -c "from lib.unified_ingest import UnifiedIngestor; print('Ingest OK')"
python -c "from lib.search.hybrid_search import HybridSearcher; print('Search OK')"

# Test schema
psql -U polymath -d polymath -f schema/001_core.sql
psql -U polymath -d polymath -f schema/002_concepts.sql
psql -U polymath -d polymath -f schema/003_code.sql
psql -U polymath -d polymath -f schema/004_skills.sql
psql -U polymath -d polymath -f schema/006_advanced.sql
```

---

## COMMIT STRATEGY

1. **Commit 1:** "fix: Resolve critical import and class name issues"
2. **Commit 2:** "fix: Align schema with code column names"
3. **Commit 3:** "fix: Unify database connections and thread safety"
4. **Commit 4:** "chore: Add missing dependencies and tests"
5. **Commit 5:** "docs: Update ARCHITECTURE.md to match reality"

Push to GitHub after each commit group.

---

## Key Files Reference

| File | Purpose | Issues |
|------|---------|--------|
| `lib/config.py` | Central config | Missing PG_POOL_* |
| `lib/db/postgres.py` | DB pool | Wrong psycopg version |
| `lib/embeddings/bge_m3.py` | Embedder | Wrong class name |
| `lib/unified_ingest.py` | Main ingestion | Column names, API mismatch, thread safety |
| `lib/ingest/asset_detector.py` | Asset detection | API expects List[Dict] |
| `lib/search/hybrid_search.py` | Search | Wrong import |
| `scripts/ingest_pdf.py` | CLI | Duplicates unified_ingest |
| `scripts/github_ingest.py` | GitHub CLI | Wrong column names, hardcoded paths |
| `scripts/extract_skills.py` | Skill extraction | Wrong column names |
| `schema/001_core.sql` | Core tables | Missing columns |
| `schema/003_code.sql` | Code tables | Column name mismatches |
| `schema/006_advanced.sql` | Views | References wrong columns |

Start with Critical Fix #1 and work through in order. Ask if anything is unclear.
