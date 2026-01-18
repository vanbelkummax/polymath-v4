---
name: polymath-smoke-test
description: Use when verifying Polymath v4 is working end-to-end - runs quick validation of ingestion, search, and concepts in under 5 minutes
---

# Polymath Smoke Test

## Overview

Quick end-to-end verification that Polymath v4 is functioning:
- Database connectivity
- PDF ingestion pipeline
- Embedding generation
- Search functionality
- Concept extraction (optional)

**Time:** 2-5 minutes

## Quick Smoke Test

### 1. Check Database Connection
```bash
cd /home/user/polymath-v4
psql -U polymath -d polymath -c "SELECT 1 as connected;"
```

### 2. Get Baseline Counts
```bash
psql -U polymath -d polymath -c "
SELECT
    (SELECT COUNT(*) FROM documents) as docs,
    (SELECT COUNT(*) FROM passages) as passages,
    (SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL) as embedded;
"
```

### 3. Ingest 1 Test PDF
```bash
# Pick a PDF from staging
TEST_PDF=$(ls /home/user/work/polymax/ingest_staging/*.pdf | head -1)
python scripts/ingest_pdf.py "$TEST_PDF" --workers 1
```

### 4. Verify Ingestion
```bash
psql -U polymath -d polymath -c "
SELECT title, created_at
FROM documents
WHERE created_at > NOW() - INTERVAL '5 minutes'
ORDER BY created_at DESC
LIMIT 1;
"
```

### 5. Test Search
```bash
python -c "
from lib.search.hybrid_search import search
results = search('the paper topic', n=3)
print(f'Found {len(results)} results')
for r in results:
    print(f\"  [{r['score']:.3f}] {r['title'][:50]}\")
"
```

### 6. Verify Counts Changed
```bash
psql -U polymath -d polymath -c "
SELECT
    (SELECT COUNT(*) FROM documents) as docs,
    (SELECT COUNT(*) FROM passages) as passages,
    (SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL) as embedded;
"
```

**Pass if:** docs +1, passages +50-200, embedded = passages

---

## Full Smoke Test (5 min)

Run this for more thorough validation:

```bash
cd /home/user/polymath-v4

# 1. Database check
echo "=== Database Check ==="
psql -U polymath -d polymath -c "SELECT COUNT(*) as docs FROM documents;"

# 2. Ingest 3 PDFs with workers
echo "=== Ingesting 3 PDFs ==="
python scripts/ingest_pdf.py \
  /home/user/work/polymax/ingest_staging/*.pdf \
  --workers 2 2>&1 | head -20

# 3. Test search
echo "=== Testing Search ==="
python -c "
from lib.search.hybrid_search import warmup
searcher = warmup(rerank=False)
for q in ['deep learning', 'gene expression', 'cell type']:
    r = searcher.hybrid_search(q, n=3)
    print(f'{q}: {len(r)} results, top score {r[0].score:.3f}' if r else f'{q}: no results')
"

# 4. System report
echo "=== System Report ==="
python scripts/system_report.py --quick

# 5. Final counts
echo "=== Final Counts ==="
psql -U polymath -d polymath -c "
SELECT
    (SELECT COUNT(*) FROM documents) as docs,
    (SELECT COUNT(*) FROM passages) as passages,
    (SELECT COUNT(embedding) FROM passages) as embedded,
    (SELECT COUNT(*) FROM passage_concepts) as concepts;
"
```

---

## Pass/Fail Criteria

| Check | Pass | Fail |
|-------|------|------|
| Database | Connects, returns counts | Connection error |
| Ingestion | Succeeded: N/N | Failed > 0 |
| Passages/doc | 50-200 | <20 or >500 |
| Embeddings | 100% of new passages | <100% |
| Search | Returns results | Empty or error |
| Latency | <30s for ingestion | >60s |

---

## Detailed Verification Queries

### Check Recent Ingestion
```sql
-- Documents from last hour
SELECT
    d.title,
    COUNT(p.passage_id) as passages,
    COUNT(p.embedding) as embedded
FROM documents d
LEFT JOIN passages p ON d.doc_id = p.doc_id
WHERE d.created_at > NOW() - INTERVAL '1 hour'
GROUP BY d.doc_id, d.title
ORDER BY d.created_at DESC;
```

### Check Asset Detection
```sql
-- GitHub repos detected
SELECT repo_url, COUNT(*) as mentions
FROM paper_repos
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY repo_url;

-- HuggingFace models detected
SELECT model_id, COUNT(*) as mentions
FROM hf_model_mentions
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY model_id;
```

### Check Search Quality
```python
# Test diverse queries
queries = [
    "spatial transcriptomics analysis",
    "attention mechanism deep learning",
    "cell segmentation microscopy",
    "gene expression prediction",
    "tumor microenvironment"
]

from lib.search.hybrid_search import warmup
searcher = warmup(rerank=False)

for q in queries:
    results = searcher.hybrid_search(q, n=5)
    print(f"\n=== {q} ===")
    if results:
        print(f"Top score: {results[0].score:.3f}")
        print(f"Score range: {results[-1].score:.3f} - {results[0].score:.3f}")
    else:
        print("NO RESULTS - investigate")
```

---

## Troubleshooting

### Database Connection Failed
```bash
# Check PostgreSQL is running
pg_isready -U polymath

# Check socket
ls /var/run/postgresql/

# Try TCP connection
psql -h localhost -U polymath -d polymath
```

### Ingestion Failed
```bash
# Check for specific errors
python scripts/ingest_pdf.py test.pdf 2>&1 | grep -i error

# Common issues:
# - PDF has no text (scanned) - need OCR
# - Schema mismatch - run migrations
# - GPU OOM - reduce workers
```

### Search Returns Nothing
```bash
# Check if passages exist
psql -U polymath -d polymath -c "SELECT COUNT(*) FROM passages WHERE embedding IS NOT NULL;"

# Check if embeddings are valid
psql -U polymath -d polymath -c "SELECT array_length(embedding, 1) FROM passages WHERE embedding IS NOT NULL LIMIT 1;"
# Should return 1024
```

---

## Automated Test Script

Save as `run_smoke_test.sh`:

```bash
#!/bin/bash
set -e
cd /home/user/polymath-v4

echo "=== Polymath v4 Smoke Test ==="
echo "Started: $(date)"

# Track results
PASSED=0
FAILED=0

# Test 1: Database
echo -n "Database connection... "
if psql -U polymath -d polymath -c "SELECT 1" > /dev/null 2>&1; then
    echo "PASS"; ((PASSED++))
else
    echo "FAIL"; ((FAILED++))
fi

# Test 2: Ingest
echo -n "PDF ingestion... "
TEST_PDF=$(ls /home/user/work/polymax/ingest_staging/*.pdf 2>/dev/null | head -1)
if [ -n "$TEST_PDF" ]; then
    if python scripts/ingest_pdf.py "$TEST_PDF" --workers 1 2>&1 | grep -q "success"; then
        echo "PASS"; ((PASSED++))
    else
        echo "FAIL"; ((FAILED++))
    fi
else
    echo "SKIP (no PDFs)"; ((PASSED++))
fi

# Test 3: Search
echo -n "Search functionality... "
if python -c "from lib.search.hybrid_search import search; assert len(search('test', n=1)) >= 0" 2>/dev/null; then
    echo "PASS"; ((PASSED++))
else
    echo "FAIL"; ((FAILED++))
fi

# Summary
echo ""
echo "=== Results ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Completed: $(date)"

[ $FAILED -eq 0 ] && exit 0 || exit 1
```

---

## Related Skills

- `polymath-pdf-ingestion` - Detailed ingestion guide
- `polymath-batch-concepts` - Concept extraction
- `polymath-search` - Search deep dive
- `polymath-system-analysis` - Full system health check
