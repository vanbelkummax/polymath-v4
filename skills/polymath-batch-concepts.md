---
name: polymath-batch-concepts
description: Use when extracting concepts from passages using Gemini batch API - handles job submission, status monitoring, and result processing with 50% cost savings
---

# Polymath Batch Concept Extraction

## Overview

Extract scientific concepts from passages using Google's Gemini batch API:
- 50% cost reduction vs real-time API
- Extracts: methods, problems, domains, datasets, entities
- Confidence scores for each concept
- Async processing (submit and check later)

## Quick Reference

```bash
cd /home/user/polymath-v4

# Check how many passages need concepts
python scripts/batch_concepts.py

# Submit a batch job (50-1000 passages recommended)
python scripts/batch_concepts.py --submit --limit 100

# Check job status
python scripts/batch_concepts.py --status

# Process completed jobs
python scripts/batch_concepts.py --process
```

## Workflow

### 1. Check Pending Passages
```bash
python scripts/batch_concepts.py
# Output: Passages pending: 6,664
```

### 2. Submit Batch Job

| Batch Size | Cost | Time | Use Case |
|------------|------|------|----------|
| 50 | ~$0.005 | 5-10 min | Testing |
| 500 | ~$0.05 | 15-30 min | Regular sync |
| 2000 | ~$0.20 | 1-2 hours | Bulk processing |

```bash
# Start with small batch to verify
python scripts/batch_concepts.py --submit --limit 100
```

### 3. Monitor Progress
```bash
# Check status
python scripts/batch_concepts.py --status

# Or check in GCP Console
# https://console.cloud.google.com/vertex-ai/batch-predictions
```

### 4. Process Results
```bash
# When job shows 'SUCCEEDED'
python scripts/batch_concepts.py --process
```

### 5. Verify Extraction
```sql
-- Check concept coverage
SELECT
    COUNT(DISTINCT p.passage_id) as total_passages,
    COUNT(DISTINCT pc.passage_id) as with_concepts,
    ROUND(100.0 * COUNT(DISTINCT pc.passage_id) / COUNT(DISTINCT p.passage_id), 1) as coverage
FROM passages p
LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id;

-- Concept distribution by type
SELECT concept_type, COUNT(*) as count
FROM passage_concepts
GROUP BY concept_type
ORDER BY count DESC;

-- Top concepts
SELECT concept_name, concept_type, COUNT(*) as occurrences
FROM passage_concepts
GROUP BY concept_name, concept_type
ORDER BY occurrences DESC
LIMIT 20;
```

## Expected Output

### Submission
```
2026-01-17 18:09:00 - INFO - Uploaded 50 requests to gs://polymath-batch-jobs/batch_input/concepts_xxx.jsonl
2026-01-17 18:09:01 - INFO - Submitted batch job: projects/xxx/batchPredictionJobs/xxx
2026-01-17 18:09:01 - INFO - Initial state: JobState.JOB_STATE_PENDING

Submitted job: projects/xxx/batchPredictionJobs/xxx
Use --status to check progress
```

### Status Check
```
Job ID                              Passages   Status    Submitted
---------------------------------------------------------------------------
projects/xxx/batchPredictionJobs/x  100        running   2026-01-17T18:08:58
```

## Configuration

Required in `.env`:
```bash
GCP_PROJECT=your-project-id
GCP_LOCATION=us-central1
GCS_BUCKET=polymath-batch-jobs
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

## Troubleshooting

### "Permission denied on resource project"
**Cause:** GCP credentials not configured
**Solution:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/home/user/.gcp/service-account.json
```

### "Model does not exist"
**Cause:** Wrong model name
**Solution:** Use `gemini-2.5-flash-lite` (not preview versions)

### Job stuck in PENDING
**Cause:** GCP quota or service issue
**Solution:** Check GCP Console for errors, may need to wait or retry

## Cost Estimation

| Model | Price | Per 1000 passages |
|-------|-------|-------------------|
| gemini-2.5-flash-lite | $0.075/1M tokens | ~$0.05 |
| Batch discount | 50% off | ~$0.025 |

## Success Criteria

| Metric | Target |
|--------|--------|
| Concepts per passage | 3-10 avg |
| Coverage | >90% of passages |
| Processing time | ~1 min per 100 passages |

## Related Skills

- `polymath-pdf-ingestion` - Ingest PDFs first
- `polymath-search` - Search by concepts
- `polymath-smoke-test` - Full system verification
