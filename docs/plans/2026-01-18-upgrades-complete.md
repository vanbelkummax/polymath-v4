# Polymath v4 Upgrades - COMPLETE

**Date:** 2026-01-18
**Status:** ✅ All 7 priorities implemented

## Summary

| # | Feature | Script | Status |
|---|---------|--------|--------|
| 1 | CORE API Integration | `scripts/discover_papers.py` | ✅ Complete |
| 2 | GraphRAG Query Expansion | `lib/search/hybrid_search.py` | ✅ Complete |
| 3 | Active Librarian | `scripts/active_librarian.py` | ✅ Complete |
| 4 | Fine-Grained NER | `scripts/detect_software_datasets.py` | ✅ Complete |
| 5 | Code Sandbox | `scripts/verify_repos.py` | ✅ Complete |
| 6 | Multi-Paper Summarization | `scripts/summarize_papers.py` | ✅ Complete |
| 7 | Streamlit Dashboard | `dashboard/app.py` | ✅ Complete |

---

## Feature Details

### 1. CORE API Integration
- **Location:** `scripts/discover_papers.py`
- **Capabilities:**
  - Search 130M+ open access papers
  - Filter by year, topic
  - Auto-dedupe against existing corpus
  - Direct text ingestion (no PDF needed - uses CORE's fullText)
  - Rate limiting (10 req/min)

**Usage:**
```bash
python scripts/discover_papers.py "spatial transcriptomics" --year-min 2022 --auto-ingest
python scripts/discover_papers.py "cell segmentation" --limit 50 --dry-run
```

### 2. GraphRAG Query Expansion
- **Location:** `lib/search/hybrid_search.py`
- **Approach:** Uses Postgres `passage_concepts` for co-occurrence (not stale Neo4j)
- **Method:** Finds concepts co-occurring with query terms, expands BM25 search

**Usage:**
```python
from lib.search.hybrid_search import HybridSearcher
searcher = HybridSearcher()
results = searcher.hybrid_search("gene expression", graph_expand=True)
```

### 3. Active Librarian
- **Location:** `scripts/active_librarian.py`
- **Capabilities:**
  - Analyze corpus gaps (missing DOIs from citations)
  - Find influential papers via Semantic Scholar
  - Generate wishlist of papers to acquire
  - Track papers needing manual retrieval

**Usage:**
```bash
python scripts/active_librarian.py --analyze-gaps
python scripts/active_librarian.py --generate-wishlist --topics "spatial transcriptomics"
```

### 4. Fine-Grained NER
- **Location:** `scripts/detect_software_datasets.py`
- **Schema:**
  - `software_registry` - Known tools with aliases
  - `dataset_registry` - Known datasets
  - `software_mentions` - Per-passage detections
  - `dataset_mentions` - Per-passage detections
- **Detection:** Pattern-based with registry normalization

**Usage:**
```bash
python scripts/detect_software_datasets.py --scan --limit 1000
python scripts/detect_software_datasets.py --stats
```

### 5. Code Sandbox
- **Location:** `scripts/verify_repos.py`
- **Schema additions:** `verification_status`, `verification_log`, `install_success`, `tests_success`
- **Approach:** Clones repos, creates venv, installs deps, runs tests

**Usage:**
```bash
python scripts/verify_repos.py --limit 10 --language Python
python scripts/verify_repos.py --verify-repo owner/name
python scripts/verify_repos.py --stats
```

### 6. Multi-Paper Summarization
- **Location:** `scripts/summarize_papers.py`
- **LLM support:** Gemini (default), Anthropic Claude
- **Output:** Markdown with inline citations and references

**Usage:**
```bash
python scripts/summarize_papers.py --query "cell segmentation methods" --top-k 10
python scripts/summarize_papers.py --query "spatial transcriptomics" --llm anthropic
```

### 7. Streamlit Dashboard
- **Location:** `dashboard/app.py`
- **Pages:**
  - 🔍 **Search** - Unified paper+repo search with GraphRAG
  - 📊 **Dashboard** - System stats, charts, recent ingestions
  - 📥 **Discovery** - CORE API search + gap analysis
  - 📝 **Literature Review** - Generate reviews from query

**Run:**
```bash
streamlit run dashboard/app.py
```

---

## Schema Additions

```sql
-- Software/Dataset Registries
software_registry (name, canonical_name, canonical_repo, aliases, category)
dataset_registry (name, canonical_name, url, description, category)

-- Mention Detection
software_mentions (passage_id, software_name, canonical_name, confidence, context)
dataset_mentions (passage_id, dataset_name, canonical_name, confidence, context)

-- Repo Verification
repositories.verification_status
repositories.verification_log
repositories.install_success
repositories.tests_success
repositories.last_verified_at
```

---

## What's Next (Priority 8)

### Neo4j Sync
Sync Postgres v4 data to Neo4j for:
- Multi-hop graph traversal
- Citation graph visualization
- Community detection on concepts
- Visual knowledge exploration

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `python scripts/discover_papers.py "query" --auto-ingest` | Discover + ingest papers |
| `python scripts/active_librarian.py --analyze-gaps` | Find missing papers |
| `python scripts/detect_software_datasets.py --scan` | Detect tool/dataset mentions |
| `python scripts/verify_repos.py --limit 10` | Verify repo quality |
| `python scripts/summarize_papers.py --query "topic"` | Generate lit review |
| `streamlit run dashboard/app.py` | Launch web UI |
