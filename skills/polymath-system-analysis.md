---
name: polymath-system-analysis
description: Use when starting a session, reviewing system health, planning improvements, or needing to understand Polymath capabilities - provides comprehensive system state analysis and improvement recommendations
---

# Polymath System Analysis

## Overview

This meta-skill provides systematic analysis of the Polymath knowledge system to identify:
- Current capabilities and coverage
- Knowledge gaps and opportunities
- Performance bottlenecks
- Improvement priorities
- Cross-domain insights potential

## When to Use

- **Session start**: Run quick health check
- **Before major tasks**: Ensure required data is available
- **Periodic review**: Monthly comprehensive analysis
- **After ingestion**: Verify new content integrated properly
- **Problem diagnosis**: When searches return poor results

---

## Quick Health Check

Run at session start (< 2 min):

```bash
cd /home/user/polymath-v3

# Database counts
psql -U polymath -d polymath -c "
SELECT 'documents' as table_name, COUNT(*) FROM documents
UNION ALL SELECT 'passages', COUNT(*) FROM passages
UNION ALL SELECT 'concepts', COUNT(*) FROM passage_concepts
UNION ALL SELECT 'code_files', COUNT(*) FROM code_files
UNION ALL SELECT 'code_chunks', COUNT(*) FROM code_chunks
UNION ALL SELECT 'skills', COUNT(*) FROM paper_skills WHERE status = 'promoted'
UNION ALL SELECT 'skill_drafts', COUNT(*) FROM paper_skills WHERE status = 'draft'
UNION ALL SELECT 'repo_queue', COUNT(*) FROM repo_queue
"

# Neo4j node counts
docker exec polymax-neo4j cypher-shell -u neo4j -p polymathic2026 "
MATCH (n) RETURN labels(n)[0] as type, count(n) as count
ORDER BY count DESC LIMIT 10
"

# Services status
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "neo4j|chroma"
```

---

## Comprehensive System Analysis

### 1. Knowledge Coverage Analysis

```sql
-- Domain coverage (papers)
SELECT
    CASE
        WHEN title ~* 'spatial|visium|xenium|merfish' THEN 'spatial_transcriptomics'
        WHEN title ~* 'pathology|histology|wsi|h&e' THEN 'pathology'
        WHEN title ~* 'single.cell|scrna|scatac' THEN 'single_cell'
        WHEN title ~* 'transformer|attention|bert|gpt' THEN 'deep_learning'
        WHEN title ~* 'graph.neural|gnn|gcn' THEN 'graph_methods'
        ELSE 'other'
    END as domain,
    COUNT(*) as papers,
    COUNT(DISTINCT d.doc_id) as unique_docs
FROM documents d
GROUP BY domain
ORDER BY papers DESC;

-- Code coverage by domain
SELECT
    CASE
        WHEN repo_name ~* 'squidpy|spatial|visium' THEN 'spatial'
        WHEN repo_name ~* 'pathology|wsi|clam|uni' THEN 'pathology'
        WHEN repo_name ~* 'scanpy|scvi|single.cell' THEN 'single_cell'
        ELSE 'other'
    END as domain,
    COUNT(DISTINCT repo_name) as repos,
    COUNT(*) as chunks
FROM code_files cf
JOIN code_chunks cc ON cf.file_id = cc.file_id
GROUP BY domain;
```

### 2. Quality Metrics

```sql
-- Embedding coverage
SELECT
    'passages' as type,
    COUNT(*) as total,
    COUNT(embedding) as with_embedding,
    ROUND(100.0 * COUNT(embedding) / NULLIF(COUNT(*), 0), 1) as pct_coverage
FROM passages
UNION ALL
SELECT
    'code_chunks',
    COUNT(*),
    COUNT(embedding_id),
    ROUND(100.0 * COUNT(embedding_id) / NULLIF(COUNT(*), 0), 1)
FROM code_chunks;

-- Concept extraction coverage
SELECT
    extractor_version,
    COUNT(*) as concepts,
    COUNT(DISTINCT passage_id) as passages
FROM passage_concepts
GROUP BY extractor_version
ORDER BY concepts DESC;

-- Document completeness
SELECT
    COUNT(*) as total_docs,
    SUM(CASE WHEN doi IS NOT NULL THEN 1 ELSE 0 END) as with_doi,
    SUM(CASE WHEN pmid IS NOT NULL THEN 1 ELSE 0 END) as with_pmid,
    SUM(CASE WHEN year IS NOT NULL THEN 1 ELSE 0 END) as with_year
FROM documents;
```

### 3. Asset Pipeline Status

```sql
-- GitHub repo queue
SELECT status, COUNT(*) as count
FROM repo_queue
GROUP BY status;

-- Skill pipeline
SELECT status, COUNT(*) as count,
       ROUND(AVG(COALESCE(evidence_count, 0)), 1) as avg_evidence
FROM paper_skills
GROUP BY status;

-- HF models tracked
SELECT
    COUNT(*) as total,
    SUM(CASE WHEN resolved THEN 1 ELSE 0 END) as resolved,
    COUNT(DISTINCT resolved_to_model_id) as unique_models
FROM hf_model_mentions;
```

---

## Gap Analysis

### Find Missing Connections

```sql
-- Papers with methods but no code links
SELECT d.title, COUNT(pc.concept_name) as method_concepts
FROM documents d
JOIN passages p ON d.doc_id = p.doc_id
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.concept_type = 'method'
AND NOT EXISTS (
    SELECT 1 FROM repo_queue rq
    WHERE rq.context ~* d.title
)
GROUP BY d.doc_id, d.title
HAVING COUNT(pc.concept_name) >= 5
ORDER BY method_concepts DESC
LIMIT 20;

-- Concepts in papers but not in code
WITH paper_concepts AS (
    SELECT DISTINCT concept_name FROM passage_concepts
    WHERE concept_type IN ('method', 'algorithm', 'tool')
),
code_concepts AS (
    SELECT DISTINCT UNNEST(concepts) as concept FROM code_chunks
)
SELECT pc.concept_name
FROM paper_concepts pc
LEFT JOIN code_concepts cc ON LOWER(pc.concept_name) = LOWER(cc.concept)
WHERE cc.concept IS NULL
LIMIT 30;
```

### Cross-Domain Opportunities

```cypher
// Find concepts that bridge domains (Neo4j)
MATCH (p1:Passage)-[:MENTIONS]->(c:Concept)<-[:MENTIONS]-(p2:Passage)
WHERE p1.domain <> p2.domain
WITH c, collect(DISTINCT p1.domain) + collect(DISTINCT p2.domain) as domains
WHERE size(domains) >= 2
RETURN c.name, domains, size(domains) as bridge_strength
ORDER BY bridge_strength DESC
LIMIT 20
```

---

## Performance Diagnosis

### Search Quality Check

```python
# Test semantic search relevance
from lib.search.hybrid_search import HybridSearcher

searcher = HybridSearcher()
test_queries = [
    "spatial gene expression prediction from histology",
    "attention mechanism for whole slide images",
    "graph neural network for cell-cell communication",
    "variational autoencoder for single cell data",
]

for query in test_queries:
    results = searcher.search(query, n=5)
    print(f"\nQuery: {query}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title'][:60]}... (score: {r['score']:.3f})")
```

### Index Health

```sql
-- Check for missing indexes
SELECT
    schemaname, tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC
LIMIT 10;

-- Vector index status
SELECT indexname, indexdef
FROM pg_indexes
WHERE indexdef LIKE '%vector%' OR indexdef LIKE '%ivfflat%';
```

---

## Improvement Priorities

### Priority Matrix

| Priority | Category | Metric | Target | Current | Action |
|----------|----------|--------|--------|---------|--------|
| **HIGH** | Coverage | Spatial papers | 100+ | Check | Ingest more Visium/Xenium papers |
| **HIGH** | Code | Priority repos | 20+ | Check | Run github_ingest.py --queue |
| **HIGH** | Quality | Embedding coverage | 95%+ | Check | Backfill embeddings |
| **MEDIUM** | Skills | Promoted skills | 50+ | Check | Review skill drafts |
| **MEDIUM** | Metadata | DOI coverage | 80%+ | Check | DOI enrichment |
| **LOW** | Graph | Neo4j concepts | 100K+ | Check | Sync more passages |

### Quick Wins

```bash
# 1. Process pending GitHub repos
python scripts/github_ingest.py --queue --limit 5

# 2. Discover new assets from papers
python scripts/discover_assets.py --recommend --add-to-queue

# 3. Review skill drafts for promotion
python scripts/promote_skill.py --check-all

# 4. Backfill missing embeddings
python scripts/backfill_embeddings_batch.py --limit 1000

# 5. Generate updated asset registry
python scripts/generate_asset_registry.py
```

---

## System Capabilities Reference

### What Polymath Can Do Now

| Capability | How to Use | Quality |
|------------|-----------|---------|
| Semantic paper search | `hybrid_search.search("query")` | Good |
| Code search | `code_search.search("function name")` | Good |
| Concept lookup | Neo4j `MATCH (c:Concept {name: "..."})` | Good |
| Cross-paper linking | Citations + shared concepts | Medium |
| Skill extraction | `skill_extractor.extract()` | New |
| Gap detection | `discover_assets.py --gaps` | New |

### What Needs Work

| Capability | Current State | Blocker |
|------------|--------------|---------|
| Cross-domain insights | Basic concept overlap | Need BridgeAnalyzer |
| Hypothesis generation | Manual | Need LLM integration |
| Automated writing | Templates only | Need manuscript generation |
| Real-time paper alerts | None | Need RSS/API integration |

---

## Monthly Review Checklist

- [ ] Run comprehensive system analysis
- [ ] Check all service health
- [ ] Review and process skill drafts
- [ ] Update HF model reference
- [ ] Ingest top priority repos
- [ ] Run search quality tests
- [ ] Check embedding coverage
- [ ] Verify Neo4j sync status
- [ ] Update ASSET_REGISTRY.md
- [ ] Document any new capabilities

---

## Report Generation

Generate comprehensive system report:

```bash
# Full system report
python scripts/system_report.py --output SYSTEM_STATUS.md

# Quick stats for session
python scripts/system_report.py --quick
```

---

*This skill should evolve as Polymath capabilities grow. Update when adding new features.*
