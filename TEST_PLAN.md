# Polymath v4 End-to-End Test Plan

## Test Scope

| Stage | Minimum | Recommended | Stress Test |
|-------|---------|-------------|-------------|
| PDFs | 5 | 20 | 100+ |
| GitHub repos | 2 | 5 | 20 |
| Concept extraction | 50 passages | 200 passages | 1000+ |
| Search queries | 10 | 25 | 50 |

---

## Phase 1: Database Setup (5 min)

```bash
cd /home/user/polymath-v4

# 1. Initialize fresh database (or use existing polymath db)
psql -U polymath -d polymath -c "SELECT 1" || createdb -U polymath polymath

# 2. Run all migrations
for f in schema/*.sql; do
    echo "Running $f..."
    psql -U polymath -d polymath -f "$f"
done

# 3. Verify tables exist
psql -U polymath -d polymath -c "
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
"
```

**Expected:** 15+ tables including documents, passages, passage_concepts, repo_queue, paper_skills

---

## Phase 2: PDF Ingestion Test (10-30 min)

### Test Set Recommendation

Use papers from your existing collection:
```bash
# Find test PDFs (spatial transcriptomics papers recommended)
ls /home/user/work/polymax/ingest_staging/*.pdf | head -20

# Or use specific papers you know well
TEST_PDFS=(
    "/path/to/spatial_transcriptomics_review.pdf"
    "/path/to/attention_mechanism_paper.pdf"
    "/path/to/cell_segmentation_methods.pdf"
    # ... add 5-20 papers
)
```

### Run Ingestion

```bash
# Single PDF test first
python scripts/ingest_pdf.py /path/to/test.pdf -v

# Batch ingestion (5-20 PDFs)
python scripts/ingest_pdf.py /path/to/papers/*.pdf --workers 4

# Or use unified ingestor directly
python -c "
from lib.unified_ingest import ingest_directory
result = ingest_directory('/path/to/papers/', workers=4)
print(f'Ingested {result.successful}/{result.total_files} files')
print(f'Total passages: {result.passages_added}')
"
```

### Evaluation Queries

```sql
-- Check document count
SELECT COUNT(*) as docs FROM documents;
-- Expected: Number of PDFs ingested

-- Check passages created
SELECT
    COUNT(*) as total_passages,
    COUNT(embedding) as with_embeddings,
    ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as embed_pct
FROM passages;
-- Expected: 50-200 passages per paper, 100% with embeddings

-- Check for errors (documents without passages)
SELECT d.doc_id, d.title, COUNT(p.passage_id) as passages
FROM documents d
LEFT JOIN passages p ON d.doc_id = p.doc_id
GROUP BY d.doc_id, d.title
HAVING COUNT(p.passage_id) = 0;
-- Expected: Empty (all docs have passages)

-- Assets detected
SELECT
    (SELECT COUNT(*) FROM repo_queue) as github_repos,
    (SELECT COUNT(*) FROM hf_model_mentions) as hf_models;
-- Expected: Some repos/models if papers reference GitHub
```

---

## Phase 3: Concept Extraction Test (15-60 min)

### Submit Batch Job

```bash
# Check pending passages
python scripts/batch_concepts.py

# Submit batch (start small: 50-200 passages)
python scripts/batch_concepts.py --submit --limit 200

# Monitor status
python scripts/batch_concepts.py --status

# Process when complete
python scripts/batch_concepts.py --process
```

### Evaluation Queries

```sql
-- Concept extraction coverage
SELECT
    COUNT(DISTINCT p.passage_id) as total_passages,
    COUNT(DISTINCT pc.passage_id) as passages_with_concepts,
    ROUND(100.0 * COUNT(DISTINCT pc.passage_id) / COUNT(DISTINCT p.passage_id), 1) as coverage_pct
FROM passages p
LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id;
-- Expected: >90% coverage after batch processing

-- Concept distribution by type
SELECT concept_type, COUNT(*) as count
FROM passage_concepts
GROUP BY concept_type
ORDER BY count DESC;
-- Expected: methods, entities, domains, problems, datasets

-- Top concepts
SELECT concept_name, concept_type, COUNT(*) as occurrences
FROM passage_concepts
GROUP BY concept_name, concept_type
ORDER BY occurrences DESC
LIMIT 20;
-- Expected: Domain-relevant terms (attention, transformer, spatial, etc.)
```

---

## Phase 4: GitHub Ingestion Test (10-20 min)

### Queue Some Repos

```bash
# Add repos manually
python scripts/github_ingest.py https://github.com/mahmoodlab/CLAM
python scripts/github_ingest.py https://github.com/scverse/squidpy

# Or process queue from paper detections
python scripts/github_ingest.py --queue --limit 5
```

### Evaluation Queries

```sql
-- Repo status
SELECT status, COUNT(*) FROM repo_queue GROUP BY status;
-- Expected: Some 'completed', maybe some 'pending'

-- Code chunks created
SELECT
    cf.repo_name,
    COUNT(DISTINCT cf.file_id) as files,
    COUNT(DISTINCT cc.chunk_id) as chunks
FROM code_files cf
LEFT JOIN code_chunks cc ON cf.file_id = cc.file_id
GROUP BY cf.repo_name;
-- Expected: 50-500 chunks per repo
```

---

## Phase 5: Search Test (5 min)

### Test Queries

```python
from lib.search.hybrid_search import HybridSearcher

searcher = HybridSearcher(rerank=True)

# Test queries (adjust to your domain)
queries = [
    "spatial transcriptomics analysis workflow",
    "attention mechanism in transformers",
    "cell segmentation deep learning",
    "gene expression normalization",
    "image to gene prediction",
]

for q in queries:
    results = searcher.hybrid_search(q, n=5)
    print(f"\n=== {q} ===")
    for r in results[:3]:
        print(f"  [{r.score:.3f}] {r.title[:50]}...")
        print(f"           {r.passage_text[:100]}...")
```

### Evaluation Criteria

| Metric | Pass | Fail |
|--------|------|------|
| Results returned | ≥3 per query | 0 results |
| Relevance | Top result matches query intent | Completely off-topic |
| Score distribution | Scores vary (0.3-0.9) | All same score |
| Response time | <2s per query | >10s |

---

## Phase 6: Skill Extraction Test (10 min)

```bash
# Extract skills from method-rich passages
python scripts/extract_skills.py --limit 20 --verbose

# Check results
python scripts/promote_skill.py --list
```

### Evaluation Queries

```sql
-- Skills created
SELECT skill_name, skill_type, evidence_count, status
FROM paper_skills
ORDER BY created_at DESC
LIMIT 10;
-- Expected: Some skills with evidence_count >= 1

-- Skills by type
SELECT skill_type, COUNT(*) FROM paper_skills GROUP BY skill_type;
```

---

## Phase 7: Neo4j Sync Test (5 min)

```bash
# Sync to Neo4j
python scripts/sync_neo4j.py --incremental

# Verify in Neo4j
docker exec polymath-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "
MATCH (n) RETURN labels(n)[0] as type, COUNT(*) as count
ORDER BY count DESC;
"
```

---

## Complete Test Script

Save this as `run_e2e_test.sh`:

```bash
#!/bin/bash
set -e

echo "=== Polymath v4 E2E Test ==="
cd /home/user/polymath-v4

# Phase 1: Setup
echo -e "\n[1/7] Database setup..."
for f in schema/*.sql; do psql -U polymath -d polymath -f "$f" 2>/dev/null; done

# Phase 2: Ingest PDFs
echo -e "\n[2/7] Ingesting PDFs..."
python scripts/ingest_pdf.py "$1" --workers 4

# Phase 3: Concepts (async - just submit)
echo -e "\n[3/7] Submitting concept extraction..."
python scripts/batch_concepts.py --submit --limit 100

# Phase 4: GitHub (if repos detected)
echo -e "\n[4/7] Processing GitHub queue..."
python scripts/github_ingest.py --queue --limit 3 || echo "No repos in queue"

# Phase 5: Search test
echo -e "\n[5/7] Testing search..."
python -c "
from lib.search.hybrid_search import HybridSearcher
s = HybridSearcher()
r = s.hybrid_search('deep learning methods', n=3)
print(f'Search returned {len(r)} results')
for x in r: print(f'  - {x.title[:60]}')
"

# Phase 6: Skills
echo -e "\n[6/7] Extracting skills..."
python scripts/extract_skills.py --limit 10

# Phase 7: Report
echo -e "\n[7/7] Final report..."
python scripts/system_report.py --quick

echo -e "\n=== E2E Test Complete ==="
```

Run with: `bash run_e2e_test.sh /path/to/test/papers/`

---

## Success Criteria Summary

| Component | Metric | Pass Threshold |
|-----------|--------|----------------|
| Ingestion | Passages created | >50 per PDF |
| Ingestion | Embedding coverage | 100% |
| Concepts | Extraction coverage | >90% |
| Concepts | Concepts per passage | 3-10 avg |
| Search | Results returned | >0 for all queries |
| Search | Latency | <2s |
| GitHub | Chunks per repo | >50 |
| Skills | Skills created | >0 |
| Overall | No Python errors | 0 exceptions |

---

## Quick Smoke Test (2 min)

For a fast sanity check:

```bash
# 1. Ingest 1 PDF
python scripts/ingest_pdf.py /path/to/one/paper.pdf

# 2. Search for it
python -c "
from lib.search.hybrid_search import search
print(search('your paper topic', n=3))
"

# 3. Check counts
psql -U polymath -d polymath -c "
SELECT
  (SELECT COUNT(*) FROM documents) as docs,
  (SELECT COUNT(*) FROM passages) as passages,
  (SELECT COUNT(*) FROM passage_concepts) as concepts;
"
```

If all 3 work → system is functional.
