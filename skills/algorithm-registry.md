---
name: algorithm-registry
description: Build and query the Algorithm Registry - comprehensive database of algorithms with cross-domain applications, polymathic bridges, and spatial biology uses. Use for real-time algorithm lookup during hackathons or research.
allowed-tools: [Read, Bash, Grep, Glob]
---

# Algorithm Registry Skill

## Overview

The Algorithm Registry is a comprehensive database of 26,000+ algorithms extracted from papers and repos, with:
- Domain classification (topology, optimization, signal processing, etc.)
- Polymathic bridges (cross-domain transfers)
- Spatial biology applications
- OCR quality flags for math-heavy content
- Links to source papers and code repositories

## Quick Commands

```bash
cd /home/user/polymath-v4

# Search for an algorithm
python scripts/algo.py "gradient descent"
python scripts/algo.py "persistent homology"

# Browse by domain
python scripts/algo.py --domain topology
python scripts/algo.py --domain optimal_transport
python scripts/algo.py --domains          # List all domains

# Find polymathic bridges (cross-domain transfers)
python scripts/algo.py --bridges

# Algorithms for spatial biology
python scripts/algo.py --spatial

# Find algorithms for a use case
python scripts/algo.py --for "cell clustering"
python scripts/algo.py --for "gene imputation"
python scripts/algo.py --for "spatial alignment"

# Top algorithms by mentions
python scripts/algo.py --top 20

# Check OCR quality concerns
python scripts/algo.py --ocr-issues
```

## Building/Updating the Registry

When new papers are ingested, rebuild the registry:

```bash
cd /home/user/polymath-v4

# Full rebuild (all steps)
python scripts/build_algorithm_registry.py --full

# Or run individual steps:
python scripts/build_algorithm_registry.py --extract      # Extract from concepts
python scripts/build_algorithm_registry.py --link         # Link to repos
python scripts/build_algorithm_registry.py --bridges      # Find polymathic transfers
python scripts/build_algorithm_registry.py --ocr-audit    # Flag OCR issues
python scripts/build_algorithm_registry.py --spatial      # Generate spatial apps
python scripts/build_algorithm_registry.py --stats        # Show statistics

# With custom minimum mentions threshold
python scripts/build_algorithm_registry.py --extract --min-mentions 3
```

## Database Schema

```sql
-- Core tables
algorithms              -- 26K+ algorithms with domain, category, applications
algorithm_papers        -- Links to source documents
algorithm_repos         -- Links to code implementations
algorithm_bridges       -- Polymathic cross-domain transfers
algorithm_domains       -- Domain taxonomy with polymathic flags

-- Key queries
-- Find algorithms by domain
SELECT name, category, spatial_biology_uses
FROM algorithms WHERE original_domain = 'topology';

-- Find polymathic bridges
SELECT a.name, ab.source_domain, ab.target_domain
FROM algorithm_bridges ab
JOIN algorithms a ON ab.algo_id = a.algo_id
ORDER BY ab.polymathic_score DESC;

-- Algorithms with code
SELECT a.name, r.repo_url, r.stars
FROM algorithms a
JOIN algorithm_repos ar ON a.algo_id = ar.algo_id
JOIN repositories r ON ar.repo_id = r.repo_id
WHERE a.original_domain = 'graph_theory';
```

## Polymathic Domains

These domains are rich sources for cross-domain algorithm transfers:

| Domain | Spatial Biology Applications |
|--------|------------------------------|
| **topology** | Tissue architecture, tumor boundaries, cell neighborhoods |
| **optimal_transport** | Spatial alignment, cell fate trajectories, batch correction |
| **control_theory** | Gene regulatory dynamics, cell fate control |
| **game_theory** | Cell competition, tumor evolution, immune dynamics |
| **information_theory** | Gene selection, spatial information quantification |
| **compressed_sensing** | Sparse gene imputation, missing data reconstruction |
| **category_theory** | Multi-modal integration, compositional models |

## OCR Quality Concerns

Math-heavy papers may have OCR extraction issues. The registry flags these:

```bash
# Check flagged algorithms
python scripts/algo.py --ocr-issues

# In SQL
SELECT name, ocr_quality_notes
FROM algorithms
WHERE ocr_quality_flag = 'suspect';
```

**When OCR issues are found:**
1. Note the algorithm and source paper
2. Consider re-ingesting with better PDF extraction (e.g., Mathpix, GROBID with math support)
3. Manually verify mathematical formulations

## Hackathon Usage

At hackathon start:
```bash
# Quick spatial algorithm lookup
python scripts/algo.py --spatial

# Find algorithm for your problem
python scripts/algo.py --for "cell segmentation"
python scripts/algo.py --for "multimodal fusion"

# Check polymathic angles
python scripts/algo.py --bridges | grep spatial
```

During implementation:
```bash
# Find code for an algorithm
python scripts/algo.py "louvain clustering"  # Shows repo links

# Find related algorithms
python scripts/algo.py --domain graph_theory
```

## Extending the Registry

### Adding New Domain Patterns

Edit `scripts/build_algorithm_registry.py`:

```python
DOMAIN_PATTERNS = {
    'new_domain': [
        r'pattern1', r'pattern2', ...
    ],
    ...
}
```

### Adding Spatial Applications

```python
spatial_mappings = {
    'new_domain': [
        'application 1',
        'application 2',
    ],
    ...
}
```

Then run:
```bash
python scripts/build_algorithm_registry.py --spatial
```

## Current Statistics

```
Total algorithms: 26,592
Classified domains: 1,438 (rest unclassified)
Polymathic bridges: 36
Spatial applications: 519 algorithms
Paper links: 14,950
OCR concerns: 2 algorithms
```

## Maintenance

After ingesting new papers:
```bash
# Quick update (just new algorithms)
python scripts/build_algorithm_registry.py --extract --min-mentions 5

# Full refresh
python scripts/build_algorithm_registry.py --full

# Check stats
python scripts/build_algorithm_registry.py --stats
```
