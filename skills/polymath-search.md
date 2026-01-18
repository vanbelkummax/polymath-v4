---
name: polymath-search
description: Use when searching the Polymath knowledge base - handles hybrid search (vector + BM25), reranking, concept search, and model warmup for fast queries
---

# Polymath Search

## Overview

Search the Polymath knowledge base using:
- **Hybrid search**: Combines vector similarity (BGE-M3) with BM25 full-text
- **Reranking**: Cross-encoder refinement for top results
- **Concept search**: Find passages by extracted concepts
- **Warmup**: Pre-load models for fast repeated queries

## Quick Reference

### Python API

```python
from lib.search.hybrid_search import HybridSearcher, warmup, search

# Quick search (convenience function)
results = search("spatial transcriptomics methods", n=10)

# With warmup for fast repeated queries
searcher = warmup(rerank=True)  # ~6s to warm up
results = searcher.hybrid_search("query", n=10)  # ~7s per query

# Without reranking (faster)
searcher = HybridSearcher(rerank=False)
results = searcher.vector_search("query", n=10)
```

### CLI (via Python)

```bash
cd /home/user/polymath-v4

# Quick search
python -c "
from lib.search.hybrid_search import search
for r in search('spatial transcriptomics', n=5):
    print(f\"[{r['score']:.3f}] {r['title'][:50]}\")
    print(f\"         {r['text'][:80]}...\")
"
```

## Search Modes

### 1. Hybrid Search (Recommended)
Combines vector and BM25 with Reciprocal Rank Fusion:

```python
searcher = HybridSearcher(rerank=True)
results = searcher.hybrid_search(
    query="attention mechanism transformer",
    n=20,
    vector_weight=0.7,  # 70% vector, 30% BM25
    rerank=True
)
```

### 2. Vector Search Only
Pure semantic similarity:

```python
results = searcher.vector_search("gene expression prediction", n=20)
```

### 3. BM25 Search Only
Keyword/exact match:

```python
results = searcher.bm25_search("EGFR mutation", n=20)
```

### 4. Concept Search
Find passages by extracted concepts:

```python
# All concept types
results = searcher.concept_search("attention mechanism", n=20)

# Specific type
results = searcher.concept_search(
    "spatial transcriptomics",
    concept_type="domain",
    n=20
)
```

## Warmup for Performance

First query loads models (~100s without warmup). Use warmup at startup:

```python
from lib.search.hybrid_search import warmup, get_searcher

# At application startup
searcher = warmup(rerank=True)  # ~6s

# All subsequent queries are fast (~7s)
results = searcher.hybrid_search("query 1")
results = searcher.hybrid_search("query 2")

# Or use get_searcher() which warms up if needed
searcher = get_searcher()
```

## Result Object

```python
@dataclass
class SearchResult:
    passage_id: str      # UUID
    passage_text: str    # Full passage text
    doc_id: str          # Parent document UUID
    title: str           # Document title
    score: float         # Relevance score (0-1)
    source: str          # 'vector', 'bm25', 'hybrid', 'reranked', 'concept'
```

## Performance Benchmarks

| Mode | First Query | Subsequent | Notes |
|------|-------------|------------|-------|
| No warmup | ~100s | ~8s | Model loading on first |
| With warmup | ~6s | ~7s | Consistent performance |
| No rerank | ~2s | ~2s | Skip cross-encoder |

## SQL Search (Alternative)

For simple searches without Python:

```sql
-- Vector similarity search
SELECT
    p.passage_text,
    d.title,
    1 - (p.embedding <=> (
        SELECT embedding FROM passages
        WHERE passage_text ILIKE '%your query%'
        LIMIT 1
    )) as similarity
FROM passages p
JOIN documents d ON p.doc_id = d.doc_id
WHERE p.embedding IS NOT NULL
ORDER BY similarity DESC
LIMIT 10;

-- Full-text search
SELECT
    p.passage_text,
    d.title,
    ts_rank(to_tsvector('english', p.passage_text),
            plainto_tsquery('english', 'spatial transcriptomics')) as rank
FROM passages p
JOIN documents d ON p.doc_id = d.doc_id
WHERE to_tsvector('english', p.passage_text)
   @@ plainto_tsquery('english', 'spatial transcriptomics')
ORDER BY rank DESC
LIMIT 10;
```

## Example Queries

```python
# Domain-specific
searcher.hybrid_search("single cell RNA sequencing analysis pipeline")
searcher.hybrid_search("histopathology image classification deep learning")
searcher.hybrid_search("spatial gene expression prediction from H&E")

# Method-focused
searcher.hybrid_search("attention mechanism implementation")
searcher.hybrid_search("graph neural network cell type annotation")

# Problem-focused
searcher.hybrid_search("batch effect correction single cell")
searcher.hybrid_search("cell type deconvolution spatial data")
```

## Troubleshooting

### Slow First Query
**Cause:** Model loading
**Solution:** Use `warmup()` at startup

### No Results
**Cause:** Query too specific or no matching content
**Solution:**
- Try broader terms
- Use BM25 for exact matches
- Check if content is in the database

### Low Relevance Scores
**Cause:** Semantic mismatch
**Solution:**
- Try different phrasing
- Use concept search for extracted terms
- Increase `n` and manually filter

## Success Criteria

| Metric | Target |
|--------|--------|
| Latency (with warmup) | <10s |
| Top-1 relevance | Match query intent |
| Score distribution | 0.3-0.9 range |

## Related Skills

- `polymath-pdf-ingestion` - Add content to search
- `polymath-batch-concepts` - Enable concept search
- `polymath-smoke-test` - Verify search works
