---
name: polymath-cross-domain-bridging
description: Bridge algorithms and concepts from mathematics, physics, economics, and computer science into spatial biology problems. Find analogous methods across fields, identify conceptual imports that could solve spatial transcriptomics challenges, and generate novel hypotheses through cross-domain reasoning.
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch]
---

# Polymath Cross-Domain Bridging

## Overview

The most powerful insights often come from importing concepts from distant fields. This skill systematically bridges methods from mathematics, physics, economics, and CS into spatial biology problems.

## When to Use This Skill

Use this skill when:
- Stuck on a spatial analysis problem (look for analogous solutions elsewhere)
- Seeking novel approaches beyond standard bioinformatics
- Writing grants/papers that emphasize innovation
- Exploring "what methods from field X could help with problem Y?"
- Generating hypotheses through conceptual transfer

## Cross-Domain Mapping Table

| Spatial Biology Problem | Analogous Field | Borrowed Concept | Example Application |
|------------------------|-----------------|------------------|---------------------|
| Cell type deconvolution | Signal processing | Blind source separation | ICA for mixed spots |
| Spatial alignment | Computer vision | Optimal transport | Tangram, PASTE |
| Tissue architecture | Algebraic topology | Persistent homology | TDA for tumor boundaries |
| Gene regulatory networks | Control theory | Feedback systems | Stability analysis |
| Cell competition | Game theory | Evolutionary dynamics | Tumor-immune interactions |
| Sparse reconstruction | Compressed sensing | L1 minimization | Imputing missing genes |
| Multi-modal integration | Category theory | Functors/sheaves | Compositional data fusion |
| Expression dynamics | Statistical physics | Diffusion processes | Spatial smoothing |
| Niche identification | Economics | Market segmentation | Cell neighborhood clustering |
| Trajectory inference | Dynamical systems | Attractor landscapes | Waddington landscape |

## Workflow 1: Problem → Method Discovery

Given a spatial biology problem, find methods from other fields:

```bash
cd /home/user/polymath-v4

# 1. Describe your problem abstractly
PROBLEM="reconstructing gene expression from sparse spatial measurements"

# 2. Search for analogous problems across fields
python -c "
from lib.search.hybrid_search import search

# Abstract the problem
queries = [
    'sparse reconstruction from incomplete measurements',
    'compressed sensing signal recovery',
    'matrix completion low rank',
    'imputation missing data',
    'inpainting image reconstruction'
]

for q in queries:
    print(f'=== {q} ===')
    for r in search(q, n=3):
        print(f'  {r.title}: {r.text[:150]}...')
    print()
"
```

### Cross-Domain Query Templates

```python
# Physics analogies
search("diffusion equation spatial propagation")
search("statistical mechanics gene expression")
search("phase transition cell fate")

# Economics/Game theory
search("Nash equilibrium cell competition")
search("utility maximization cell behavior")
search("market dynamics tumor microenvironment")

# Control theory
search("feedback control gene regulation")
search("stability analysis biological networks")
search("optimal control trajectory")

# Information theory
search("mutual information spatial correlation")
search("channel capacity gene expression")
search("entropy spatial organization")

# Topology/Geometry
search("persistent homology tissue structure")
search("sheaf theory multi-modal integration")
search("manifold learning single cell")
```

## Workflow 2: Method → Application Discovery

Given a method from another field, find spatial biology applications:

```bash
# Example: Optimal Transport
python -c "
from lib.search.hybrid_search import search

method = 'optimal transport'
contexts = [
    f'{method} spatial transcriptomics',
    f'{method} cell alignment',
    f'{method} gene expression',
    f'{method} single cell',
    f'{method} tissue'
]

for q in contexts:
    results = search(q, n=5)
    if results:
        print(f'=== {q} ===')
        for r in results:
            print(f'  {r.title}')
"
```

## Workflow 3: Concept Genealogy

Trace how a concept traveled across fields:

```sql
-- Find the journey of "attention mechanism"
SELECT d.title, d.year,
       array_agg(DISTINCT pc.concept_name) FILTER (WHERE pc.concept_type = 'field') as fields
FROM documents d
JOIN passages p ON d.doc_id = p.doc_id
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE EXISTS (
    SELECT 1 FROM passage_concepts pc2
    WHERE pc2.passage_id = p.passage_id
    AND pc2.concept_name ILIKE '%attention%'
)
GROUP BY d.doc_id, d.title, d.year
ORDER BY d.year ASC
LIMIT 20;
```

## Polymathic Fields Reference

### High-Value Imports for Spatial Biology

**Optimal Transport** (3,830 mentions in KB)
- Already well-represented
- Applications: Tangram, PASTE, CellRank

**Information Theory** (1,235 mentions)
- Mutual information for spatial correlation
- Channel capacity for cell communication
- Entropy for tissue organization

**Topological Data Analysis** (69 mentions - EXPAND)
- Persistent homology for tissue boundaries
- Mapper for trajectory visualization
- Betti numbers for architecture

**Sheaf Theory** (~100 mentions - EXPAND)
- Multi-modal data fusion
- Consistent local-to-global inference
- Cellular sheaves for spatial networks

**Game Theory** (9 mentions - EXPAND)
- Tumor-immune interactions
- Cell competition dynamics
- Evolutionary game theory

**Control Theory** (4 mentions - EXPAND)
- Gene regulatory network stability
- Feedback loop analysis
- Optimal intervention design

### Harvest Targets

Fields to actively expand in KB:
```bash
# Run these discovery queries
python scripts/discover_papers.py "topological data analysis single cell" --limit 50
python scripts/discover_papers.py "persistent homology biology" --limit 30
python scripts/discover_papers.py "game theory tumor microenvironment" --limit 30
python scripts/discover_papers.py "control theory gene regulation" --limit 30
python scripts/discover_papers.py "sheaf theory machine learning" --limit 20
python scripts/discover_papers.py "compressed sensing genomics" --limit 30
```

## Hypothesis Generation Template

When bridging fields, use this structure:

```markdown
## Cross-Domain Hypothesis

**Source Field:** [e.g., Statistical Physics]
**Source Concept:** [e.g., Phase transitions]
**Target Problem:** [e.g., Cell fate decisions in spatial context]

### Analogy
In [source field], [concept] describes [behavior].
Similarly, in spatial biology, [target phenomenon] exhibits [analogous behavior].

### Testable Prediction
If this analogy holds, we would expect:
1. [Specific prediction 1]
2. [Specific prediction 2]

### Experimental Design
To test this, we could:
- [Approach 1]
- [Approach 2]

### Prior Art
Papers connecting these concepts:
- [Paper 1]
- [Paper 2]
```

## Quick Reference: Polymathic Queries

```bash
cd /home/user/polymath-v4

# Find physics-biology bridges
psql -U polymath -d polymath -c "
SELECT d.title, d.year FROM documents d
JOIN passages p ON d.doc_id = p.doc_id
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.concept_name IN ('statistical_mechanics', 'phase_transition', 'diffusion_equation')
  AND EXISTS (
    SELECT 1 FROM passage_concepts pc2
    WHERE pc2.passage_id = p.passage_id
    AND pc2.concept_name ILIKE '%cell%'
  )
LIMIT 10;
"

# Find math-biology bridges
psql -U polymath -d polymath -c "
SELECT d.title, d.year FROM documents d
JOIN passages p ON d.doc_id = p.doc_id
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.concept_name IN ('category_theory', 'sheaf', 'homology', 'topology')
  AND EXISTS (
    SELECT 1 FROM passage_concepts pc2
    WHERE pc2.passage_id = p.passage_id
    AND pc2.concept_type = 'domain'
    AND pc2.concept_name ILIKE '%bio%'
  )
LIMIT 10;
"
```

## Expanding the Knowledge Base

When polymathic fields are underrepresented, use waterfall acquisition:

```bash
cd /home/user/polymath-v4

# Check current field coverage
python scripts/harvest_polymathic.py --list

# Harvest a specific underrepresented field (papers + repos)
python scripts/waterfall_acquire.py --polymathic --field tda

# Harvest ALL underrepresented polymathic fields
nohup python scripts/waterfall_acquire.py --polymathic --all > /tmp/polymathic_harvest.log 2>&1 &

# Check what needs manual retrieval
python scripts/waterfall_acquire.py --show-manual

# Fill citation network gaps (papers cited but missing)
python scripts/fill_citation_gaps.py --analyze
python scripts/fill_citation_gaps.py --fetch --limit 30
```

### Waterfall Acquisition Order
1. **CORE API** - Open access aggregator
2. **Unpaywall** - OA link finder by DOI
3. **arXiv** - Direct preprint access
4. **Semantic Scholar** - OA links + metadata
5. **→ Manual list** - If all fail, saved for you to retrieve

### After Harvest
New papers automatically trigger:
- GitHub repo discovery from paper text
- Repo README/docstring indexing
- Concept extraction (via batch job)

### Manual Retrieval List
Papers that couldn't be auto-acquired are saved to:
`/home/user/polymath-v4/data/manual_retrieval_needed.jsonl`

## Integration with Other Skills

- **polymath-research-synthesis**: After bridging, synthesize findings
- **hypothesis-generation**: Formalize cross-domain hypotheses
- **polymath-exam-prep**: Practice explaining analogies (PQE gold!)
