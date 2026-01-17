# Polymath v4 Architecture

> **Lean, execution-ready knowledge system for scientific papers and code.**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              POLYMATH v4 PIPELINE                               │
│                                                                                 │
│  INPUTS           PROCESSING              STORAGE            OUTPUTS            │
│  ──────           ──────────              ───────            ───────            │
│                                                                                 │
│  ┌─────┐     ┌──────────────────┐     ┌──────────────┐                         │
│  │ PDF │────▶│ 1. Parse + Chunk │────▶│  PostgreSQL  │                         │
│  └─────┘     │    (PyMuPDF)     │     │  (pgvector)  │                         │
│              └────────┬─────────┘     └──────────────┘                         │
│                       │                      │                                  │
│                       ▼                      ▼                                  │
│              ┌──────────────────┐     ┌──────────────┐     ┌──────────────┐    │
│              │ 2. Embed (LOCAL) │────▶│  passages    │────▶│   Semantic   │    │
│              │    BGE-M3        │     │  .embedding  │     │   Search     │    │
│              └──────────────────┘     └──────────────┘     └──────────────┘    │
│                       │                                                         │
│                       ▼                                                         │
│              ┌──────────────────┐     ┌──────────────┐                         │
│              │ 3. Concepts (AI) │────▶│  passage_    │                         │
│              │  Gemini Flash    │     │  concepts    │                         │
│              └────────┬─────────┘     └──────────────┘                         │
│                       │                                                         │
│         ┌─────────────┼─────────────┐                                          │
│         ▼             ▼             ▼                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                                  │
│  │ 4. Assets  │ │ 5. Skills  │ │ 6. Cites   │                                  │
│  │  (Regex)   │ │ (Gemini)   │ │  (Regex)   │                                  │
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘                                  │
│        │              │              │                                          │
│        ▼              ▼              ▼                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                                  │
│  │ repo_queue │ │paper_skills│ │citation_   │                                  │
│  │ hf_models  │ │skill_drafts│ │links       │                                  │
│  └─────┬──────┘ └────────────┘ └────────────┘                                  │
│        │                                                                        │
│        ▼                                                                        │
│  ┌─────────┐     ┌──────────────────┐     ┌──────────────┐                     │
│  │ GitHub  │────▶│ 7. Code Ingest   │────▶│  code_chunks │                     │
│  │  Repo   │     │    (AST Parse)   │     │  code_files  │                     │
│  └─────────┘     └──────────────────┘     └──────────────┘                     │
│                                                  │                              │
│                                                  ▼                              │
│                                           ┌──────────────┐     ┌────────────┐  │
│                                           │ 8. Neo4j     │────▶│ Knowledge  │  │
│                                           │    Sync      │     │   Graph    │  │
│                                           └──────────────┘     └────────────┘  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## AI Model Usage Map

| Stage | Model | Location | Cost | Purpose |
|-------|-------|----------|------|---------|
| **2. Embeddings** | BGE-M3 (local) | GPU | $0 | 1024-dim vectors for search |
| **3. Concepts** | Gemini 2.5 Flash Lite | Batch API | ~$0.0001/passage | Extract methods, domains, entities |
| **5. Skills** | Gemini 2.5 Flash | Real-time | ~$0.001/skill | Extract actionable procedures |
| **Search Rerank** | BGE-reranker-v2-m3 | GPU | $0 | Improve search precision |

### Model Details

```yaml
Embeddings:
  model: BAAI/bge-m3
  dimensions: 1024
  location: Local GPU
  batch_size: 32

Concepts:
  model: gemini-2.5-flash-lite
  api: Vertex AI Batch (50% discount)
  max_output_tokens: 16384

Skills:
  model: gemini-2.5-flash
  api: Vertex AI (real-time)
  temperature: 0.1

Reranker:
  model: BAAI/bge-reranker-v2-m3
  location: Local GPU
```

---

## Pipeline Stages Detail

### Stage 1: PDF Parsing
```
Input:  PDF file
Output: ParseResult(text, pages, page_count)
Model:  None (PyMuPDF/fitz)
Script: lib/ingest/pdf_parser.py
```

### Stage 2: Chunking + Embeddings
```
Input:  Raw text
Output: List[Chunk] with embeddings
Model:  BGE-M3 (LOCAL, $0)
Script: lib/ingest/chunking.py + lib/embeddings/bge_m3.py
```

### Stage 3: Concept Extraction
```
Input:  Passage text
Output: List[Concept(name, type, confidence)]
Model:  Gemini 2.5 Flash Lite (BATCH API, 50% off)
Script: scripts/batch_concepts.py
```

### Stage 4: Asset Detection
```
Input:  Passage text
Output: GitHub repos, HF models, DOIs
Model:  None (Regex)
Script: lib/ingest/asset_detector.py
```

### Stage 5: Skill Extraction
```
Input:  Method-rich passages + concepts
Output: CANDIDATE.md files in skills_drafts/
Model:  Gemini 2.5 Flash (REAL-TIME)
Script: lib/ingest/skill_extractor.py
```

### Stage 6: Citation Extraction
```
Input:  Passage text
Output: DOI links
Model:  None (Regex)
Script: lib/ingest/asset_detector.py
```

### Stage 7: GitHub Code Ingestion
```
Input:  Repo URL from queue
Output: code_files, code_chunks tables
Model:  None (AST parsing)
Script: scripts/github_ingest.py
```

### Stage 8: Neo4j Sync
```
Input:  Postgres tables
Output: Knowledge graph
Model:  None
Script: scripts/sync_neo4j.py
```

---

## Database Schema

### PostgreSQL (pgvector)

```sql
-- Core tables
documents       -- Paper metadata (title, DOI, authors, year)
passages        -- Text chunks with embeddings (1024-dim)
passage_concepts -- Extracted concepts (method, domain, entity)

-- Code tables
code_files      -- Source files from repos
code_chunks     -- Functions, classes, methods

-- Asset tables
repo_queue      -- GitHub repos pending ingestion
hf_models       -- HuggingFace model references
citation_links  -- Paper-to-paper citations

-- Skill tables
paper_skills    -- Extracted skills (draft/promoted)
skill_usage_log -- Track skill effectiveness
```

### Neo4j (Knowledge Graph)

```cypher
(:Document)-[:HAS_PASSAGE]->(:Passage)-[:MENTIONS]->(:Concept)
(:Document)-[:CITES]->(:Document)
(:Document)-[:CITES_CODE]->(:Repository)-[:CONTAINS]->(:CodeChunk)
(:Skill)-[:DERIVED_FROM]->(:Passage)
(:Skill)-[:IMPLEMENTS]->(:Concept)
```

---

## Directory Structure

```
polymath-v4/
├── ARCHITECTURE.md          # This file
├── QUICKSTART.md            # Execution guide
├── CLAUDE.md                # Claude Code instructions
├── .env.example             # Environment template
│
├── lib/
│   ├── config.py            # Central configuration
│   ├── ingest/
│   │   ├── pipeline.py      # Main ingestion orchestrator
│   │   ├── pdf_parser.py    # PyMuPDF wrapper
│   │   ├── chunking.py      # Text chunking strategies
│   │   ├── asset_detector.py # GitHub/HF/DOI detection
│   │   ├── concept_extractor.py # Gemini concept extraction
│   │   └── skill_extractor.py   # Skill extraction to drafts
│   ├── embeddings/
│   │   └── bge_m3.py        # BGE-M3 embedder
│   ├── db/
│   │   ├── postgres.py      # Connection pool
│   │   └── neo4j.py         # Neo4j driver
│   └── search/
│       ├── hybrid_search.py # Vector + keyword search
│       └── reranker.py      # BGE reranker
│
├── scripts/
│   ├── ingest_pdf.py        # Single PDF ingestion
│   ├── batch_concepts.py    # Batch API concept extraction
│   ├── github_ingest.py     # GitHub repo ingestion
│   ├── discover_assets.py   # Find repos/models in papers
│   ├── sync_neo4j.py        # Postgres → Neo4j sync
│   ├── promote_skill.py     # Skill promotion (4 gates)
│   └── system_report.py     # Health check
│
├── schema/
│   ├── 001_core.sql         # documents, passages
│   ├── 002_concepts.sql     # passage_concepts
│   ├── 003_code.sql         # code_files, code_chunks, repo_queue
│   ├── 004_skills.sql       # paper_skills, skill_usage_log
│   └── 005_neo4j.cypher     # Graph constraints
│
├── skills/                  # Promoted skills (production)
│   └── .gitkeep
│
└── skills_drafts/           # Draft skills (pending review)
    └── .gitkeep
```

---

## Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION SEQUENCE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

PHASE 1: SETUP (Once)
═══════════════════════════════════════════════════════════════════════════════
  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ 1. Create .env  │────▶│ 2. Run schema   │────▶│ 3. Start Neo4j  │
  │                 │     │    migrations   │     │    container    │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  Commands:
  $ cp .env.example .env && vim .env
  $ psql -U polymath -f schema/001_core.sql
  $ docker-compose up -d neo4j


PHASE 2: PAPER INGESTION (Per Batch)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ 1. Ingest PDFs  │────▶│ 2. Batch API    │────▶│ 3. Process      │
  │   (+ embeddings)│     │   (concepts)    │     │   results       │
  │                 │     │                 │     │                 │
  │ MODEL: BGE-M3   │     │ MODEL: Gemini   │     │ MODEL: None     │
  │ COST: $0        │     │ COST: ~$0.05    │     │ COST: $0        │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  Commands:
  $ python scripts/ingest_pdf.py /path/to/papers/*.pdf
  $ python scripts/batch_concepts.py --submit
  $ python scripts/batch_concepts.py --process


PHASE 3: ASSET DISCOVERY (After Ingestion)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ 1. Discover     │────▶│ 2. Ingest       │────▶│ 3. Extract      │
  │   assets        │     │   GitHub repos  │     │   skills        │
  │                 │     │                 │     │                 │
  │ MODEL: None     │     │ MODEL: None     │     │ MODEL: Gemini   │
  │ COST: $0        │     │ COST: $0        │     │ COST: ~$0.10    │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  Commands:
  $ python scripts/discover_assets.py --github --add-to-queue
  $ python scripts/github_ingest.py --queue --limit 20
  $ python scripts/extract_skills.py --recent


PHASE 4: GRAPH SYNC (Periodic)
═══════════════════════════════════════════════════════════════════════════════

  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
  │ 1. Sync to      │────▶│ 2. Build        │────▶│ 3. Generate     │
  │   Neo4j         │     │   relationships │     │   reports       │
  │                 │     │                 │     │                 │
  │ MODEL: None     │     │ MODEL: None     │     │ MODEL: None     │
  └─────────────────┘     └─────────────────┘     └─────────────────┘

  Commands:
  $ python scripts/sync_neo4j.py --incremental
  $ python scripts/system_report.py --full
```

---

## Cost Estimation

| Operation | Model | Per Unit | 1000 Papers |
|-----------|-------|----------|-------------|
| Embeddings | BGE-M3 (local) | $0 | $0 |
| Concepts | Gemini Batch | $0.0001/passage | ~$5 |
| Skills | Gemini Real-time | $0.001/skill | ~$2 |
| Reranking | BGE-reranker (local) | $0 | $0 |
| **Total** | | | **~$7** |

---

## Skills Integration

### Skill Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SKILL LIFECYCLE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

  EXTRACTION                VALIDATION               PROMOTION
  ──────────                ──────────               ─────────

  ┌─────────────┐     ┌─────────────────────────────────────────┐     ┌───────┐
  │ Gemini      │────▶│           4 VALIDATION GATES            │────▶│skills/│
  │ extracts    │     │                                         │     │       │
  │ skill from  │     │ ┌─────────┐ ┌─────────┐ ┌─────┐ ┌─────┐│     │SKILL  │
  │ passages    │     │ │Evidence │→│ Oracle  │→│Dedup│→│Usage││     │.md    │
  │             │     │ │ ≥2 srcs │ │test pass│ │<0.85│ │ ≥1  ││     │       │
  └─────────────┘     │ └─────────┘ └─────────┘ └─────┘ └─────┘│     └───────┘
        │             └─────────────────────────────────────────┘
        ▼
  ┌─────────────┐
  │skills_drafts│
  │/CANDIDATE.md│
  └─────────────┘
```

### Available Skills

| Skill | Trigger | Stage |
|-------|---------|-------|
| `polymath-system-analysis` | Session start | System |
| `spatial-data-loading` | Load Visium/Xenium | Analysis |
| `scanpy-analysis` | Single-cell workflow | Analysis |
| `systematic-debugging` | Any bug | Development |

---

## Environment Variables

```bash
# Required
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Optional
GITHUB_TOKEN=ghp_xxx  # For higher rate limits
HF_TOKEN=hf_xxx       # For private models
```

---

## Quick Reference Commands

```bash
# Health check
python scripts/system_report.py --quick

# Ingest papers
python scripts/ingest_pdf.py paper.pdf

# Batch concepts (50% cheaper)
python scripts/batch_concepts.py --submit --limit 1000
python scripts/batch_concepts.py --status
python scripts/batch_concepts.py --process

# GitHub ingestion
python scripts/github_ingest.py https://github.com/owner/repo
python scripts/github_ingest.py --user mahmoodlab
python scripts/github_ingest.py --queue --limit 10

# Discovery
python scripts/discover_assets.py --recommend

# Skills
python scripts/promote_skill.py --list
python scripts/promote_skill.py skill-name --bootstrap

# Search
python -c "from lib.search.hybrid_search import search; print(search('query'))"
```
