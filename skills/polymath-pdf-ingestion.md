---
name: polymath-pdf-ingestion
description: Use when ingesting PDFs into Polymath v4 - handles single files, batch ingestion with parallel workers, and asset detection (GitHub repos, HF models, citations)
---

# Polymath PDF Ingestion

## Overview

Ingest scientific papers (PDFs) into the Polymath knowledge base with:
- Text extraction and chunking
- BGE-M3 embeddings (1024-dim)
- Asset detection (GitHub repos, HuggingFace models, citations)
- Deduplication via title hash

## Quick Reference

```bash
cd /home/user/polymath-v4

# Single PDF
python scripts/ingest_pdf.py /path/to/paper.pdf

# Multiple PDFs with parallel workers
python scripts/ingest_pdf.py /path/to/papers/*.pdf --workers 4

# Skip embeddings (faster, for testing)
python scripts/ingest_pdf.py paper.pdf --no-embeddings

# Skip asset detection
python scripts/ingest_pdf.py paper.pdf --no-assets
```

## Recommended Workflow

### 1. Stage PDFs
```bash
# Copy PDFs to staging directory
cp ~/Downloads/*.pdf /home/user/work/polymax/ingest_staging/
```

### 2. Preview What Will Be Ingested
```bash
ls /home/user/work/polymax/ingest_staging/*.pdf | wc -l
```

### 3. Ingest with Appropriate Worker Count

| PDF Count | Workers | Notes |
|-----------|---------|-------|
| 1-5 | 1 | Single-threaded is fine |
| 5-20 | 2 | Parallel but GPU-bound |
| 20+ | 4 | Max practical parallelism |

```bash
# For 10 PDFs
python scripts/ingest_pdf.py /home/user/work/polymax/ingest_staging/*.pdf --workers 2
```

### 4. Verify Ingestion
```sql
-- Check recent documents
SELECT title, created_at
FROM documents
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;

-- Check passage counts
SELECT d.title, COUNT(p.passage_id) as passages
FROM documents d
JOIN passages p ON d.doc_id = p.doc_id
WHERE d.created_at > NOW() - INTERVAL '1 hour'
GROUP BY d.doc_id, d.title;

-- Check embedding coverage
SELECT
    COUNT(*) as total,
    COUNT(embedding) as with_embeddings,
    ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as pct
FROM passages p
JOIN documents d ON p.doc_id = d.doc_id
WHERE d.created_at > NOW() - INTERVAL '1 hour';
```

## Expected Output

```
2026-01-17 18:05:26,583 - INFO - Starting batch 'ingest_20260117_180526' with 5 files, 2 workers
2026-01-17 18:05:32,495 - INFO - Parsing: paper1.pdf
2026-01-17 18:05:33,760 - INFO - Computing embeddings for 73 chunks...
2026-01-17 18:05:34,729 - INFO - Stored 1 GitHub repos for doc xxx
2026-01-17 18:05:34,742 - INFO - Ingested: Paper Title... (73 passages)

============================================================
BATCH SUMMARY
============================================================
Total: 5
Succeeded: 5
Failed: 0
Time: 45.2s
```

## Troubleshooting

### "Cannot copy out of meta tensor" Error
**Cause:** Race condition in multi-worker mode (fixed in v4)
**Solution:** Use `--workers 1` or ensure you have the latest code with threading locks

### "No text extracted" Error
**Cause:** PDF is scanned/image-only
**Solution:** Use OCR preprocessing (not yet in v4, use v3's OCR pipeline)

### Slow Ingestion
**Cause:** GPU bottleneck or large PDFs
**Solution:**
- Reduce worker count
- Check GPU memory with `nvidia-smi`
- Process in smaller batches

## Success Criteria

| Metric | Target |
|--------|--------|
| Passages per PDF | 50-200 |
| Embedding coverage | 100% |
| Processing time | ~10s per PDF |
| Failed files | 0 |

## Related Skills

- `polymath-batch-concepts` - Extract concepts from ingested passages
- `polymath-search` - Search the ingested content
- `polymath-smoke-test` - Full system verification
