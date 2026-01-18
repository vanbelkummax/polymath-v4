# Repository Indexing Design

**Date:** 2026-01-18
**Status:** Approved
**Author:** Claude + Max

## Overview

Add GitHub repositories as first-class entities in Polymath v4, enabling search across both academic papers AND implementation code.

## Goals

1. Index repos with README + Python docstrings
2. Link repos to papers when detected
3. Extract concepts from repo content (unified with paper concepts)
4. Enable queries like "find repos implementing spatial autocorrelation"

## Scope

| Source | Count | Notes |
|--------|-------|-------|
| Paper-linked repos | ~1,277 | Detected from paper text |
| Orphaned repos | ~1,019 | From archived data |
| Curated list | ~30 | Key spatial/ML repos |
| **Total** | ~2,300 | Before dedup/filtering |

## Content Extraction

Per repo:
- README.md (1-3 passages)
- Python docstrings from .py files (5-20 passages)
- Estimated: 10-25 passages per repo average

## Schema

```sql
-- Core repo table
CREATE TABLE repositories (
    repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url TEXT UNIQUE NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,

    -- GitHub metadata
    description TEXT,
    language TEXT,
    stars INT,
    forks INT,
    topics TEXT[],
    default_branch TEXT DEFAULT 'main',

    -- Content
    readme_content TEXT,

    -- Tracking
    source_method TEXT,  -- 'paper_detection', 'curated', 'orphaned'
    indexed_at TIMESTAMP DEFAULT NOW(),
    last_github_sync TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Repo passages (parallel to paper passages)
CREATE TABLE repo_passages (
    passage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,

    passage_text TEXT NOT NULL,
    section TEXT,           -- 'readme', 'docstring', 'module_doc'
    file_path TEXT,         -- e.g., 'src/analysis.py'
    function_name TEXT,     -- e.g., 'calculate_moran'

    embedding vector(1024),
    quality_score REAL,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Paper-repo links (bidirectional)
CREATE TABLE paper_repo_links (
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_id UUID NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,

    link_type TEXT,         -- 'mentioned_in_paper', 'implements', 'uses'
    confidence REAL,
    detected_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (doc_id, repo_id)
);

-- Indexes
CREATE INDEX idx_repo_passages_embedding ON repo_passages
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX idx_repo_passages_repo_id ON repo_passages(repo_id);
CREATE INDEX idx_repositories_owner_name ON repositories(owner, name);
CREATE INDEX idx_paper_repo_links_repo ON paper_repo_links(repo_id);
```

## Ingestion Pipeline

```
1. COLLECT REPO URLs
   ├── Query paper_repos table (1,277)
   ├── Query archive.paper_repos_orphaned (1,019)
   ├── Add curated list (~30)
   └── Deduplicate by URL

2. FETCH GITHUB METADATA
   ├── GET /repos/{owner}/{repo}
   ├── Extract: stars, forks, language, topics, description
   └── Rate limit: 5,000/hour with token

3. FETCH README
   ├── GET /repos/{owner}/{repo}/readme
   ├── Base64 decode
   └── Store raw markdown

4. FETCH PYTHON DOCSTRINGS
   ├── GET /repos/{owner}/{repo}/git/trees/{branch}?recursive=1
   ├── Filter: *.py files
   ├── GET raw content for each .py file
   ├── Parse with ast module for docstrings
   └── Extract: module, class, function docstrings

5. CHUNK & EMBED
   ├── Chunk README (~500 chars)
   ├── Chunk docstrings (one per function/class)
   ├── BGE-M3 embeddings
   └── Store in repo_passages

6. LINK TO PAPERS
   ├── Match by URL in paper_repos
   ├── Insert into paper_repo_links
   └── Set link_type based on context
```

## Curated Spatial/ML Repos

Priority repos to include even if not paper-detected:

```
# Spatial Transcriptomics
scverse/squidpy
scverse/spatialdata
theislab/scanpy
satijalab/seurat
mouseland/cellpose
stardist/stardist

# ST Prediction (H&E → expression)
mahmoodlab/HIPT
mahmoodlab/CLAM
owkin/HistoSSL
katherlab/HIA

# Foundation Models
huggingface/transformers
facebookresearch/dinov2
google/gemma
openai/CLIP

# ML Infrastructure
pytorch/pytorch
numpy/numpy
pandas-dev/pandas
scikit-learn/scikit-learn
```

## False Positive Filtering

Skip repos matching:
- `github.com/blog`
- `github.com/posts`
- `github.com/about`
- Single-word paths without owner (e.g., `github.com/picard`)
- Repos returning 404

## Performance Estimates

| Step | Time | Notes |
|------|------|-------|
| GitHub metadata fetch | ~30 min | 2,300 repos @ 5,000/hr |
| README fetch | ~30 min | Same rate |
| Python files fetch | ~2 hrs | ~10 files/repo average |
| Embedding | ~1 hr | ~50K passages @ 100/sec |
| **Total** | ~4 hrs | Can run in background |

## Search Integration

Update `hybrid_search.py` to search both:

```python
def search(query, n=10, include_repos=True):
    paper_results = search_papers(query, n)
    if include_repos:
        repo_results = search_repos(query, n)
        return merge_and_rerank(paper_results, repo_results, n)
    return paper_results
```

## Concept Extraction

After repo ingestion, run batch concept extraction on:
- All paper passages (~143K)
- All repo passages (~50K estimated)

Same Gemini batch API, same concept schema, unified concept graph.
