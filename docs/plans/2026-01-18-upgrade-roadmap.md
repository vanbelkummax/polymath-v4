# Polymath v4 - Major Upgrades Session

## Current State (2026-01-18)

### Database
| Table | Count | Notes |
|-------|-------|-------|
| `documents` | 1,698 | Papers with BGE-M3 embeddings |
| `passages` | 143,103 | Paper passages |
| `repositories` | ~2,000 | GitHub repos (README + docstrings) |
| `repo_passages` | ~30,000 | Repo passages with embeddings |
| `paper_repo_links` | TBD | Bidirectional links |

### Key Files
| File | Purpose |
|------|---------|
| `/home/user/polymath-v4/` | Main codebase |
| `/home/user/polymath-v4/.env` | API keys (CORE_API_KEY added) |
| `/home/user/polymath-v4/lib/search/hybrid_search.py` | Search implementation |
| `/home/user/polymath-v4/scripts/batch_concepts.py` | Gemini concept extraction |
| `/home/user/polymath-v4/scripts/ingest_pdf.py` | PDF ingestion |
| `/home/user/polymath-v4/scripts/ingest_repos.py` | Repo ingestion |
| `/home/user/CLAUDE.md` | Full project context |

### APIs Available
| API | Key Location | Purpose |
|-----|--------------|---------|
| **CORE API** | `.env` CORE_API_KEY | 130M+ open access papers |
| **Gemini** | GCP service account | Batch concept extraction |
| **GitHub** | GITHUB_TOKEN | Repo metadata/content |
| **Neo4j** | `.env` NEO4J_PASSWORD | Concept graph |

---

## UPGRADES TO IMPLEMENT

### 1. CORE API Integration (Paper Discovery)

**Goal:** Auto-discover papers by topic instead of manual Zotero import.

**Create `scripts/discover_papers.py`:**
```python
# Endpoints:
# - https://api.core.ac.uk/v3/search/works?q=QUERY&limit=100
# - Header: Authorization: Bearer {CORE_API_KEY}

# Features:
# - Search by topic: "spatial transcriptomics cell segmentation"
# - Filter by year: yearPublished>=2020
# - Auto-download PDFs when available
# - Dedupe against existing documents by DOI
# - Feed into ingest_pdf.py pipeline
```

**Usage:**
```bash
python scripts/discover_papers.py "spatial transcriptomics" --year-min 2020 --limit 50
python scripts/discover_papers.py "graph neural networks biology" --auto-ingest
```

**CORE API Examples:**
```
GET /v3/search/works?q=title:"spatial transcriptomics"&limit=10
GET /v3/search/works?q=fullText:"Visium" AND yearPublished>=2022
```

---

### 2. GraphRAG (Query Expansion via Neo4j)

**Goal:** Smarter search by expanding queries with related concepts from the knowledge graph.

**Modify `lib/search/hybrid_search.py`:**
```python
def _expand_query_with_graph(self, query: str) -> str:
    """Expand query using Neo4j concept neighbors."""
    # 1. Extract concepts from query (simple NER or exact match)
    # 2. Query Neo4j for related concepts:
    #    MATCH (c:Concept {name: $concept})-[:RELATED_TO]-(neighbor)
    #    RETURN neighbor.name LIMIT 5
    # 3. Append to query: "spatial transcriptomics OR Visium OR squidpy"
    pass
```

**Example:**
- Input: "spatial transcriptomics"
- Neo4j returns: ["Visium", "Squidpy", "MERFISH", "cell segmentation"]
- Expanded: "spatial transcriptomics OR Visium OR Squidpy OR MERFISH"

---

### 3. Code Sandbox (Repo Verification)

**Goal:** Automatically verify if repos actually run.

**Update schema:**
```sql
ALTER TABLE repositories ADD COLUMN verification_status TEXT DEFAULT 'unverified';
-- Values: 'unverified', 'verified', 'failed', 'no_tests'
ALTER TABLE repositories ADD COLUMN verification_log TEXT;
ALTER TABLE repositories ADD COLUMN last_verified_at TIMESTAMP;
```

**Create `scripts/verify_repos.py`:**
```python
# For each repo where verification_status = 'unverified':
# 1. Clone to temp directory
# 2. Try: pip install -r requirements.txt (or setup.py)
# 3. Try: pytest or python -m pytest
# 4. Record success/failure + logs
# 5. Update database
```

**Usage:**
```bash
python scripts/verify_repos.py --limit 50 --language Python
```

---

### 4. Active Librarian (Auto-Find Missing Papers)

**Goal:** Automatically discover and ingest frequently-cited papers missing from corpus.

**Create `scripts/active_research.py`:**
```python
# 1. Query citation_links for DOIs cited >3 times but not in documents
# 2. Use Semantic Scholar API to get metadata + PDF link
# 3. Use CORE API as fallback for open access version
# 4. Auto-ingest via ingest_pdf.py
```

**Usage:**
```bash
python scripts/active_research.py --min-citations 3 --auto-ingest
```

---

### 5. Fine-Grained NER (Software/Dataset Detection)

**Goal:** Detect tool and dataset mentions without hyperlinks.

**Modify `scripts/batch_concepts.py`:**

Update the Gemini prompt to extract:
```json
{
  "concepts": [...],
  "software_mentions": ["Scanpy", "Seurat", "CellRanger"],
  "dataset_mentions": ["TCGA", "GTEx", "HPA"]
}
```

**Create lookup tables:**
```sql
CREATE TABLE software_registry (
    name TEXT PRIMARY KEY,
    canonical_repo TEXT,  -- e.g., "github.com/theislab/scanpy"
    aliases TEXT[]        -- ["scanpy", "sc.pp", "scanpy.pp"]
);

CREATE TABLE dataset_registry (
    name TEXT PRIMARY KEY,
    url TEXT,
    description TEXT
);
```

---

### 6. Multi-Paper Summarization

**Goal:** Generate literature review summaries from multiple papers.

**Create `scripts/summarize_papers.py`:**
```python
# 1. Take list of doc_ids or search query
# 2. Retrieve top passages from each paper
# 3. Send to Gemini/Claude with prompt:
#    "Synthesize these passages into a literature review section on {topic}"
# 4. Output markdown with citations
```

**Usage:**
```bash
python scripts/summarize_papers.py --query "cell segmentation methods" --top-k 10
python scripts/summarize_papers.py --doc-ids uuid1,uuid2,uuid3
```

---

### 7. Streamlit Dashboard

**Goal:** Visual interface for non-CLI users.

**Create `dashboard/app.py`:**
```python
import streamlit as st
# Pages:
# - Search (papers + repos unified)
# - Paper Discovery (CORE API search)
# - Knowledge Graph visualization
# - Ingestion status monitor
# - Literature review generator
```

**Run:**
```bash
streamlit run dashboard/app.py
```

---

## Implementation Order

| Priority | Feature | Complexity | Impact |
|----------|---------|------------|--------|
| 1 | CORE API Integration | Medium | High |
| 2 | GraphRAG | Medium | High |
| 3 | Active Librarian | Low | Medium |
| 4 | Fine-Grained NER | Low | Medium |
| 5 | Code Sandbox | Medium | Medium |
| 6 | Multi-Paper Summarization | Low | High |
| 7 | Streamlit Dashboard | Medium | Medium |

---

## Quick Start Commands

```bash
cd /home/user/polymath-v4

# Check current status
bash scripts/status.sh

# Test CORE API
curl -H "Authorization: Bearer $(grep CORE_API_KEY .env | cut -d= -f2)" \
  "https://api.core.ac.uk/v3/search/works?q=spatial+transcriptomics&limit=3"

# Current search
python -c "from lib.search.hybrid_search import search; print(search('spatial transcriptomics', n=3))"

# Check Neo4j
cypher-shell -u neo4j -p polymathic2026 "MATCH (n) RETURN count(n)"
```

---

## Architecture After Upgrades

```
                    ┌─────────────────┐
                    │   CORE API      │
                    │  (Discovery)    │
                    └────────┬────────┘
                             │
                             ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Papers    │────▶│    Polymath     │◀────│   Repos     │
│  (1,698)    │     │   PostgreSQL    │     │  (~2,000)   │
└─────────────┘     └────────┬────────┘     └─────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Neo4j    │  │ Hybrid   │  │ Gemini   │
        │ GraphRAG │  │ Search   │  │ Concepts │
        └──────────┘  └──────────┘  └──────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Streamlit     │
                    │   Dashboard     │
                    └─────────────────┘
```

---

## Notes

- CORE API key is in `/home/user/polymath-v4/.env`
- Rate limit: 10 requests/minute free tier
- GraphRAG requires concepts in Neo4j (run batch_concepts.py first)
- Code sandbox needs Docker for safe execution (optional)
