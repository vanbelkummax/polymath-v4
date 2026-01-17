# Polymath v4 - Claude Code Guide

## What This Is

Polymath v4 is a lean knowledge system that:
- Ingests scientific papers (PDFs)
- Extracts concepts using Gemini
- Discovers and ingests GitHub repositories
- Builds a searchable knowledge graph
- Extracts actionable skills from literature

---

## Quick Commands

```bash
# Health check
python scripts/system_report.py --quick

# Ingest papers
python scripts/ingest_pdf.py paper.pdf
python scripts/ingest_pdf.py /path/to/papers/*.pdf --workers 4

# Extract concepts (batch API, 50% cheaper)
python scripts/batch_concepts.py --submit --limit 1000
python scripts/batch_concepts.py --status
python scripts/batch_concepts.py --process

# Discover assets from papers
python scripts/discover_assets.py --github --add-to-queue

# Ingest GitHub repos
python scripts/github_ingest.py https://github.com/owner/repo
python scripts/github_ingest.py --queue --limit 10

# Sync to Neo4j
python scripts/sync_neo4j.py --incremental

# Skills
python scripts/promote_skill.py --list
python scripts/promote_skill.py skill-name --bootstrap
```

---

## AI Models Used

| Stage | Model | Cost |
|-------|-------|------|
| Embeddings | BGE-M3 (local GPU) | $0 |
| Concepts | Gemini 2.5 Flash Lite (batch) | ~$0.0001/passage |
| Skills | Gemini 2.5 Flash | ~$0.001/skill |
| Reranking | BGE-reranker-v2-m3 (local) | $0 |

---

## Directory Structure

```
polymath-v4/
├── lib/                 # Core library
│   ├── config.py        # Configuration
│   ├── ingest/          # Ingestion modules
│   ├── embeddings/      # BGE-M3 embedder
│   ├── db/              # Database connections
│   └── search/          # Hybrid search
├── scripts/             # CLI tools
├── schema/              # SQL + Cypher migrations
├── skills/              # Promoted skills
└── skills_drafts/       # Draft skills
```

---

## Database

```bash
# Connect to Postgres
psql -U polymath -d polymath

# Key tables
documents       # Paper metadata
passages        # Text chunks + embeddings
passage_concepts # Extracted concepts
code_files      # GitHub source files
code_chunks     # Functions/classes
repo_queue      # Pending repos
paper_skills    # Skills
```

---

## Verification

```sql
-- Document stats
SELECT COUNT(*) as docs,
       (SELECT COUNT(*) FROM passages) as passages,
       (SELECT COUNT(*) FROM passage_concepts) as concepts
FROM documents;

-- Embedding coverage
SELECT
  COUNT(*) as total,
  COUNT(embedding) as embedded,
  ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as pct
FROM passages;

-- Queue status
SELECT status, COUNT(*) FROM repo_queue GROUP BY status;
```

---

## Architecture

See `ARCHITECTURE.md` for complete system design.

Key flows:
1. PDF → Parse → Chunk → Embed → Store
2. Passages → Gemini Batch → Concepts
3. Passages → Asset Detection → repo_queue
4. repo_queue → Clone → AST Parse → code_chunks
5. All → Neo4j Sync → Knowledge Graph

---

## Skills

Skills are extracted from papers and validated through 4 gates:
1. **Evidence**: ≥2 source passages
2. **Oracle**: Test passes
3. **Dedup**: <0.85 similarity to existing
4. **Usage**: ≥1 logged success

Draft skills live in `skills_drafts/`, promoted skills in `skills/`.

---

## Environment

Required in `.env`:
```
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```
