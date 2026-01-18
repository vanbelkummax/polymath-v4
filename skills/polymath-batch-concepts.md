---
name: polymath-batch-concepts
description: Use when extracting concepts from passages using Gemini batch API - handles job submission, status checking, and result processing with 50% cost savings
---

# Polymath Batch Concept Extraction

## Overview

Extract scientific concepts from passages using Gemini's batch API:
- **50% cost savings** vs synchronous API
- **5 concept types**: methods, problems, domains, datasets, entities
- **~18 concepts per passage** average yield
- **~5 minutes** for 50 passages

## Quick Reference

```bash
cd /home/user/polymath-v4

# Check pending passages
python scripts/batch_concepts.py

# Submit batch job (50-500 passages recommended)
python scripts/batch_concepts.py --submit --limit 100

# Check job status
python scripts/batch_concepts.py --status

# Process completed results
python scripts/batch_concepts.py --process
```

## Workflow

### 1. Check What's Pending

```bash
python scripts/batch_concepts.py
```

Output:
```
Concept extraction status:
  Total concepts: 902
  Passages pending: 1,234

Use --submit to start extraction
```

### 2. Submit Batch Job

```bash
# Recommended batch sizes
python scripts/batch_concepts.py --submit --limit 50   # Quick test (~5 min)
python scripts/batch_concepts.py --submit --limit 500  # Standard batch (~30 min)
python scripts/batch_concepts.py --submit --limit 1000 # Large batch (~1 hour)
```

Output:
```
Found 100 passages to process
Uploaded 100 requests to gs://polymath-batch-jobs/batch_input/concepts_YYYYMMDD_HHMMSS.jsonl
Submitted batch job: projects/.../batchPredictionJobs/1234567890
Initial state: JOB_STATE_PENDING

Use --status to check progress
```

### 3. Monitor Job Status

```bash
python scripts/batch_concepts.py --status
```

Output:
```
Job ID                                             Passages   Status       Submitted
--------------------------------------------------------------------------------------------
projects/.../batchPredictionJobs/5556735143177093120 50         succeeded    2026-01-17T18:28:08
```

**Status values:**
- `running` - Job in progress
- `succeeded` - Ready to process
- `failed` - Check GCP console for errors
- `processed` - Results already imported

### 4. Process Results

```bash
python scripts/batch_concepts.py --process
```

Output:
```
Processing results from gs://polymath-batch-jobs/batch_input/.../dest
Loaded mapping for 50 passages
Added 902 concepts from job projects/.../5556735143177093120
```

### 5. Verify Results

```sql
-- Check concept distribution
SELECT concept_type, COUNT(*), ROUND(AVG(confidence)::numeric, 2) as avg_conf
FROM passage_concepts
WHERE extractor_version = 'batch-v4'
GROUP BY concept_type
ORDER BY COUNT(*) DESC;

-- Top concepts by type
SELECT concept_name, COUNT(*) as mentions
FROM passage_concepts
WHERE concept_type = 'method' AND extractor_version = 'batch-v4'
GROUP BY concept_name
ORDER BY mentions DESC
LIMIT 20;
```

## Concept Types

| Type | Description | Examples |
|------|-------------|----------|
| **method** | Techniques, algorithms, procedures | attention mechanism, gradient descent, SCTransform |
| **problem** | Research challenges | cell type classification, batch effect correction |
| **domain** | Research areas | spatial transcriptomics, computational pathology |
| **dataset** | Data sources | TCGA, ImageNet, Visium HD |
| **entity** | Tools, diseases, genes, proteins | PyTorch, GBM, TP53, EGFR |

## Tested Performance (50 passages)

| Metric | Observed |
|--------|----------|
| Processing time | ~5 minutes |
| Concepts extracted | 902 |
| Concepts/passage | ~18 |
| Avg confidence | 0.84-0.91 |

### Distribution (Real Results)
```
 concept_type | count | avg_conf
--------------+-------+----------
 entity       |   234 |     0.91
 method       |   210 |     0.88
 problem      |   206 |     0.87
 domain       |   139 |     0.84
 dataset      |   109 |     0.89
```

## Architecture

```
submit_batch_job()
├── Create JSONL with prompts
├── Upload to GCS: batch_input/concepts_YYYYMMDD_HHMMSS.jsonl
├── Save mapping: batch_input/concepts_YYYYMMDD_HHMMSS_mapping.json
├── Submit to Gemini batch API (google.genai SDK)
└── Track in concept_batch_jobs table

process_results()
├── Get job from Gemini API (job.dest.gcs_uri)
├── Load passage mapping (line index → passage_id)
├── Parse predictions.jsonl
├── Strip ```json markdown blocks
└── INSERT into passage_concepts
```

## GCS Structure

```
gs://polymath-batch-jobs/
├── batch_input/
│   ├── concepts_20260117_182813.jsonl         # Input requests
│   ├── concepts_20260117_182813_mapping.json  # passage_id mapping
│   └── concepts_20260117_182813/
│       └── dest/
│           └── prediction-model-.../
│               └── predictions.jsonl          # Output (created by Gemini)
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

### Job stuck in "running"

```bash
# Check actual status from GCP
python -c "
from google import genai
from lib.config import config

client = genai.Client(vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION)
job = client.batches.get(name='projects/.../batchPredictionJobs/JOB_ID')
print(f'State: {job.state}')
print(f'Stats: {job.completion_stats}')
"
```

### "No completed jobs to process"

```bash
# Check job status in database
psql -U polymath -d polymath -c "SELECT job_id, status FROM concept_batch_jobs ORDER BY submitted_at DESC LIMIT 5;"

# If succeeded in GCP but not DB, update manually
psql -U polymath -d polymath -c "UPDATE concept_batch_jobs SET status = 'succeeded' WHERE job_id = 'JOB_ID';"
```

### 0 concepts added

**Fixed in current version.** Previous issues were:
1. Wrong output path - now uses `job.dest.gcs_uri`
2. Missing passage mapping - uses `*_mapping.json` file
3. Markdown blocks - now strips ```json blocks
4. Wrong ON CONFLICT - now includes `extractor_version`

### Job failed

```bash
python -c "
from google import genai
from lib.config import config

client = genai.Client(vertexai=True, project=config.GCP_PROJECT, location=config.GCP_LOCATION)
job = client.batches.get(name='JOB_ID')
print(f'Error: {job.error}')
"
```

Common failures:
- JSONL format (fixed: now includes "request" wrapper)
- Model name (use `gemini-2.5-flash-lite`)
- Quota exceeded

### "Permission denied on resource project"

```bash
# Check .env is loaded
python -c "from lib.config import config; print(config.GCP_PROJECT)"

# Should print your project ID, not placeholder
```

## Database Schema

```sql
-- Job tracking
CREATE TABLE concept_batch_jobs (
    job_id TEXT PRIMARY KEY,
    gcs_input_uri TEXT,
    gcs_output_uri TEXT,
    passage_count INTEGER,
    status TEXT,  -- pending, running, succeeded, failed, processed
    submitted_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- Concept storage (PK includes extractor_version)
CREATE TABLE passage_concepts (
    passage_id UUID NOT NULL,
    concept_name TEXT NOT NULL,
    concept_type TEXT,  -- method, problem, domain, dataset, entity
    confidence REAL,
    extractor_model TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (passage_id, concept_name, extractor_version)
);
```

## Success Criteria

| Metric | Target | Observed |
|--------|--------|----------|
| Job completes | < 10 min for 50 | ✓ 5 min |
| Concepts/passage | 10-25 | ✓ 18 |
| Avg confidence | > 0.80 | ✓ 0.84-0.91 |
| Parse success | > 95% | ✓ 100% |

## Related Skills

- `polymath-pdf-ingestion` - Ingest PDFs before concept extraction
- `polymath-search` - Search by concepts
- `polymath-smoke-test` - Verify system health
