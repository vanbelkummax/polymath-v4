---
name: polymath-fresh-ingest
description: Use when running the fresh v4 database build with Zotero integration, deduplication, and LaTeX preservation - the full three-stage pipeline
---

# Polymath Fresh Ingest Pipeline

## Overview

Full pipeline for building the v4 knowledge base from scratch with:
- Zotero CSV integration for rich metadata
- Strict deduplication (DOI + title_hash + year + authors)
- LaTeX preservation for math equations
- Three-stage architecture (ingest → enrich → concepts)

**Design Doc:** `/home/user/polymath-v4/docs/plans/2026-01-17-fresh-ingest-pipeline-design.md`

## Data Sources

| Source | Location | Count |
|--------|----------|-------|
| Zotero Library | `/mnt/c/Users/User/Downloads/My Library.csv` | 5,348 entries |
| Staging PDFs | `/home/user/work/polymax/ingest_staging/` | 1,400 PDFs |
| Existing Repos | `paper_repos` table | 1,603 (preserve!) |

## Stage 1: Ingest (Tonight)

### Step 1.1: Prepare Zotero CSV

```bash
cd /home/user/polymath-v4

# Deduplicate and prepare Zotero metadata
python scripts/prepare_zotero_ingest.py '/mnt/c/Users/User/Downloads/My Library.csv' \
    --output /home/user/work/polymax/ingest_staging/zotero_prepared.csv
```

**Expected output:**
- Removes duplicate DOIs
- Maps Windows paths to Linux
- Creates lookup table for PDF matching

### Step 1.2: Run Batch Ingest

```bash
# Run in background (4+ hours)
cd /home/user/polymath-v4
nohup python scripts/ingest_pdf.py /home/user/work/polymax/ingest_staging/ \
    --workers 2 \
    --zotero-csv /home/user/work/polymax/ingest_staging/zotero_prepared.csv \
    --batch-name fresh_build_2026_01_17 \
    --recursive \
    > /home/user/logs/ingest_fresh_build.log 2>&1 &

echo $! > /home/user/logs/ingest_fresh_build.pid
```

### Step 1.3: Monitor Progress

```bash
# Watch log
tail -f /home/user/logs/ingest_fresh_build.log

# Check counts
psql -U polymath -d polymath -c "
SELECT
    ingest_batch,
    COUNT(*) as docs,
    COUNT(DISTINCT doi) as with_doi
FROM documents
WHERE ingest_batch LIKE 'fresh%'
GROUP BY ingest_batch;
"
```

## Stage 2: Enrich (Background)

Run after Stage 1 completes:

```bash
cd /home/user/polymath-v4

# CrossRef enrichment (papers with DOI)
python scripts/enrich_metadata.py --source crossref --limit 500

# OpenAlex enrichment (citations, concepts)
python scripts/enrich_metadata.py --source openalex --limit 500

# Semantic Scholar (influential citations)
python scripts/enrich_metadata.py --source s2 --limit 500
```

## Stage 3: Concepts (Batch Job)

```bash
cd /home/user/polymath-v4

# Submit batch to Gemini
python scripts/batch_concepts.py --submit --limit 1000

# Check status
python scripts/batch_concepts.py --status

# Process results when done
python scripts/batch_concepts.py --process
```

## Deduplication Rules

**Strict approach:** Only dedupe when certain

| Match Type | Duplicate? |
|------------|------------|
| Same DOI | Yes |
| Same PDF hash | Yes |
| Same title_hash + year + overlapping authors | Yes |
| Same title, different DOI | No (different papers) |
| Same title, different authors | No |
| Same title, different year | No |

## Verification Queries

```sql
-- Overall stats
SELECT
    COUNT(*) as total_docs,
    COUNT(doi) as with_doi,
    COUNT(abstract) as with_abstract,
    ROUND(100.0 * COUNT(doi) / COUNT(*), 1) as doi_pct
FROM documents
WHERE ingest_batch LIKE 'fresh%';

-- Check for duplicates
SELECT doi, COUNT(*) as cnt
FROM documents
WHERE doi IS NOT NULL
GROUP BY doi
HAVING COUNT(*) > 1;

-- Embedding coverage
SELECT
    COUNT(*) as passages,
    COUNT(embedding) as embedded,
    ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as pct
FROM passages p
JOIN documents d ON p.doc_id = d.doc_id
WHERE d.ingest_batch LIKE 'fresh%';

-- Asset detection results
SELECT
    (SELECT COUNT(*) FROM paper_repos) as repos,
    (SELECT COUNT(*) FROM hf_models) as hf_models,
    (SELECT COUNT(*) FROM citation_links) as citations;
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Documents ingested | 1,400+ |
| DOI coverage | >50% |
| Embedding coverage | 100% |
| Repos preserved | 1,603+ |
| Zero true duplicates | DOI unique constraint holds |

## Troubleshooting

### Zotero paths don't match
```bash
# Check path mapping
head -5 /home/user/work/polymax/ingest_staging/zotero_prepared.csv
# Verify PDFs exist at mapped paths
```

### Memory issues during batch
```bash
# Reduce workers
python scripts/ingest_pdf.py ... --workers 1
```

### Duplicate constraint violation
```bash
# Check existing doc
psql -U polymath -d polymath -c "SELECT doc_id, title FROM documents WHERE doi = 'YOUR_DOI';"
```

## Related Skills

- `polymath-pdf-ingestion` - Basic PDF ingestion (single/batch)
- `polymath-batch-concepts` - Concept extraction
- `polymath-search` - Search verification
- `polymath-smoke-test` - Full system test
