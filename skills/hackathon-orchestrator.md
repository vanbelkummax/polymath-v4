---
name: hackathon-orchestrator
description: INTERNAL SKILL for Claude to orchestrate spatial multimodal hackathon. Speed-optimized queries, decision trees, and polymathic patterns. Load this at hackathon start.
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch, Task]
---

# Hackathon Orchestrator (Claude Internal)

## LOAD AT HACKATHON START

```bash
cd /home/user/polymath-v4

# 1. Full warmup (caches models + common queries) - ~60s
python scripts/hackathon_warmup.py

# 2. Load knowledge pack
cat data/hackathon_knowledge_pack.json | python -c "import sys,json; d=json.load(sys.stdin); print(f'Loaded: {len(d[\"architectures\"])} architecture categories')"
```

## SPEED TOOLS

```bash
# Quick search (2-3s per query)
python scripts/q.py "your query"
python scripts/q.py "your query" -n 20
python scripts/q.py "your query" --fast    # Skip reranking (~1s)
python scripts/q.py "your query" --repos   # Search code repos
python scripts/q.py "your query" --code    # Find code for papers

# Use case lookup (INSTANT)
python scripts/usecase.py                   # List all use cases
python scripts/usecase.py gene              # Gene expression prediction
python scripts/usecase.py multimodal        # Multimodal integration
python scripts/usecase.py --decisions       # Quick decision trees
python scripts/usecase.py --emergency       # Emergency fixes
```

## PRE-MADE USE CASES

| Use Case | Likelihood | Command |
|----------|------------|---------|
| Gene expression prediction | 🔴 HIGH | `usecase.py gene` |
| Multimodal integration | 🔴 HIGH | `usecase.py multimodal` |
| Benchmark evaluation | 🔴 HIGH | `usecase.py benchmark` |
| Spatial domain ID | 🟡 MED | `usecase.py spatial` |
| Foundation model tuning | 🟡 MED | `usecase.py foundation` |
| Cell type deconvolution | 🟡 MED | `usecase.py deconv` |
| Cell segmentation | 🟢 LOW | `usecase.py segment` |

---

## PHASE 1: PROBLEM SCOPING (First 30 min)

### Understand the Challenge
When user describes hackathon challenge, immediately run:

```python
# 1. What exists already?
from lib.search.hybrid_search import search
challenge_keywords = "USER_CHALLENGE_KEYWORDS"
existing = search(f"{challenge_keywords} deep learning", n=15)

# 2. What are the SOTA baselines?
baselines = search(f"{challenge_keywords} benchmark state-of-the-art", n=10)

# 3. What datasets exist?
datasets = search(f"{challenge_keywords} dataset public available", n=10)
```

### Quick SOTA Check (Run in parallel)
```sql
-- Find papers with code for this problem
SELECT d.title, d.year, pr.repo_url, r.stars
FROM documents d
JOIN paper_repos pr ON d.doc_id = pr.doc_id
JOIN repositories r ON LOWER(pr.repo_url) = LOWER(r.repo_url)
WHERE d.title ILIKE '%CHALLENGE_KEYWORD%'
ORDER BY r.stars DESC NULLS LAST
LIMIT 10;
```

---

## PHASE 2: ARCHITECTURE SELECTION (30-60 min)

### Decision Tree: Spatial Multimodal

```
INPUT: H&E + Spatial Coordinates + Gene Expression
                    |
    +---------------+---------------+
    |               |               |
  IMAGE         SPATIAL          MULTI-MODAL
  ENCODER       ENCODER          FUSION
    |               |               |
    v               v               v

IMAGE OPTIONS:
├── ResNet50 (fast, proven) → search("resnet histology pretrained")
├── ViT/DeiT (attention) → search("vision transformer pathology")
├── UNI/CONCH (foundation) → search("foundation model pathology UNI CONCH")
├── CTransPath (histology-specific) → search("CTransPath histology")
└── HIPT (hierarchical) → search("HIPT hierarchical image pyramid")

SPATIAL OPTIONS:
├── GNN (graph structure) → search("graph neural network spatial transcriptomics")
├── Positional encoding → search("positional encoding spatial coordinates")
├── Spatial attention → search("spatial attention mechanism tissue")
└── Coordinate MLP → search("coordinate network spatial")

FUSION OPTIONS:
├── Concatenation (simple) → baseline
├── Cross-attention → search("cross attention multimodal")
├── CLIP-style contrastive → search("contrastive learning histology gene")
├── Optimal transport → search("optimal transport multimodal alignment")
└── Tensor fusion → search("tensor fusion multimodal")
```

### Pre-Written Architecture Queries

```python
# Vision encoders for histology
search("pretrained vision model histology pathology", n=10)
search("self-supervised learning histology representation", n=10)

# Spatial modeling
search("graph neural network cell neighborhood spatial", n=10)
search("attention mechanism spatial context window", n=10)

# Gene expression prediction
search("gene expression prediction from histology image", n=15)
search("transcriptomics image deep learning predict", n=10)

# Multimodal fusion
search("multimodal fusion image gene expression", n=10)
search("cross-modal learning pathology transcriptomics", n=10)
```

---

## PHASE 3: IMPLEMENTATION (Main Phase)

### Find Code to Fork

```sql
-- Top starred spatial biology repos
SELECT name, stars, language, repo_url, description
FROM repositories
WHERE (description ILIKE '%spatial%' AND description ILIKE '%transcript%')
   OR name ILIKE '%spatial%'
   OR name ILIKE '%st-%' OR name ILIKE '%hist%gene%'
ORDER BY stars DESC NULLS LAST
LIMIT 20;

-- Repos with specific techniques
SELECT DISTINCT r.name, r.repo_url, r.stars
FROM repositories r
JOIN repo_passages rp ON r.repo_id = rp.repo_id
WHERE rp.passage_text ILIKE '%TECHNIQUE_NAME%'
ORDER BY r.stars DESC NULLS LAST
LIMIT 10;
```

### Common Implementation Patterns

```python
# Loss functions for gene expression
search("loss function gene expression prediction MSE Poisson", n=10)
search("negative binomial loss RNA-seq deep learning", n=5)

# Data augmentation for histology
search("data augmentation histology pathology stain", n=10)
search("color normalization histology deep learning", n=5)

# Training tricks
search("batch size learning rate histology training", n=5)
search("mixed precision training pathology", n=5)
```

---

## PHASE 4: POLYMATHIC EDGE (Use Throughout)

### Cross-Domain Imports

| Problem | Query to Run |
|---------|--------------|
| Sparse gene imputation | `search("compressed sensing matrix completion genomics", n=10)` |
| Spatial alignment | `search("optimal transport image registration", n=10)` |
| Cell neighborhood | `search("graph attention network node classification", n=10)` |
| Multi-scale features | `search("pyramid pooling hierarchical features", n=10)` |
| Uncertainty | `search("uncertainty quantification deep learning prediction", n=10)` |
| Few-shot | `search("few-shot learning domain adaptation", n=10)` |

### Physics-Inspired Patterns

```python
# Diffusion for spatial smoothing
search("diffusion model spatial denoising", n=5)

# Energy-based for cell interactions
search("energy based model spatial interaction", n=5)

# Information theory for feature selection
search("mutual information feature selection gene", n=5)
```

### Math-Inspired Patterns

```python
# Topology for tissue structure
search("persistent homology tissue architecture shape", n=5)

# Optimal transport for alignment
search("Wasserstein distance spatial alignment", n=5)

# Manifold for latent space
search("manifold learning gene expression latent", n=5)
```

---

## PHASE 5: VALIDATION & BENCHMARKS

### Find Benchmarks

```python
search("spatial transcriptomics benchmark evaluation metrics", n=10)
search("gene expression prediction correlation PCC", n=5)
search("Visium benchmark dataset evaluation", n=5)
```

### Standard Metrics (Quick Reference)

| Task | Metrics | Query |
|------|---------|-------|
| Gene prediction | PCC, MAE, RMSE | `search("gene prediction pearson correlation", n=5)` |
| Spot deconvolution | RMSE, JSD | `search("deconvolution evaluation metrics", n=5)` |
| Cell type | Accuracy, F1 | `search("cell type classification accuracy", n=5)` |
| Spatial pattern | Moran's I | `search("spatial autocorrelation Moran", n=5)` |

### Find Baselines to Beat

```sql
SELECT d.title, d.year, pr.repo_url
FROM documents d
LEFT JOIN paper_repos pr ON d.doc_id = pr.doc_id
WHERE d.title ILIKE '%HisToGene%' OR d.title ILIKE '%ST-Net%'
   OR d.title ILIKE '%Img2ST%' OR d.title ILIKE '%HEST%'
   OR d.title ILIKE '%Tangram%' OR d.title ILIKE '%cell2location%';
```

---

## QUICK COMMAND REFERENCE

### One-Liners for Speed

```bash
# Quick search (copy-paste ready)
python -c "from lib.search.hybrid_search import search; [print(f'{r.title}: {r.text[:100]}') for r in search('YOUR_QUERY', n=5)]"

# Find repos for a method
psql -U polymath -d polymath -c "SELECT name, repo_url, stars FROM repositories WHERE description ILIKE '%METHOD%' ORDER BY stars DESC LIMIT 5;"

# Find paper for a repo
psql -U polymath -d polymath -c "SELECT d.title, d.year FROM documents d JOIN paper_repos pr ON d.doc_id = pr.doc_id WHERE pr.repo_url ILIKE '%REPO_NAME%';"

# Count what we have
psql -U polymath -d polymath -c "SELECT (SELECT COUNT(*) FROM documents) as papers, (SELECT COUNT(*) FROM repositories) as repos, (SELECT COUNT(*) FROM passages) as passages;"
```

### Database Stats Query

```sql
-- Quick corpus overview
SELECT
  (SELECT COUNT(*) FROM documents WHERE year >= 2024) as recent_papers,
  (SELECT COUNT(DISTINCT concept_name) FROM passage_concepts WHERE concept_type = 'method') as methods,
  (SELECT COUNT(*) FROM repositories WHERE stars > 100) as popular_repos;
```

---

## HACKATHON-SPECIFIC SCENARIOS

### Scenario: "We need to predict gene expression from H&E"

1. **Existing methods:**
```python
search("gene expression prediction histology image deep learning", n=15)
```

2. **Architecture patterns:**
```python
search("encoder decoder gene expression image", n=10)
search("vision transformer gene prediction", n=10)
```

3. **Loss functions:**
```python
search("loss function gene expression Poisson negative binomial", n=10)
```

4. **Code to fork:**
```sql
SELECT r.name, r.repo_url, r.stars FROM repositories r
JOIN repo_passages rp ON r.repo_id = rp.repo_id
WHERE rp.passage_text ILIKE '%gene expression%' AND rp.passage_text ILIKE '%predict%'
ORDER BY r.stars DESC NULLS LAST LIMIT 10;
```

### Scenario: "We need to integrate multiple modalities"

1. **Fusion strategies:**
```python
search("multimodal fusion early late hybrid", n=10)
search("cross-modal attention transformer", n=10)
```

2. **Contrastive approaches:**
```python
search("CLIP contrastive image text biology", n=10)
search("contrastive learning multimodal representation", n=10)
```

3. **Alignment methods:**
```python
search("optimal transport multimodal alignment", n=10)
search("canonical correlation analysis multiomics", n=5)
```

### Scenario: "We need to model spatial relationships"

1. **Graph approaches:**
```python
search("graph neural network spatial cell neighborhood", n=10)
search("graph attention network tissue", n=5)
```

2. **Attention approaches:**
```python
search("spatial attention mechanism local global", n=10)
search("self-attention spatial context", n=5)
```

3. **Positional encoding:**
```python
search("positional encoding 2D coordinates spatial", n=10)
```

---

## MCP INTEGRATIONS (External Knowledge)

### Clinical Trials (if relevant)
```
mcp__clinical-trials__search_trials(condition="spatial transcriptomics", status=["RECRUITING"])
```

### ChEMBL (if drug-related)
```
mcp__chembl__target_search(gene_symbol="EGFR")
```

### Vanderbilt Professors (local expertise)
```
mcp__vanderbilt-professors__search_huo_papers(query="spatial transcriptomics")
mcp__vanderbilt-professors__search_lau_papers(query="colorectal cancer spatial")
```

---

## ERROR RECOVERY

### If search is slow
```python
# Use non-reranked search (faster)
from lib.search.hybrid_search import HybridSearcher
s = HybridSearcher(use_reranker=False)
results = s.hybrid_search("query", n=10)
```

### If need more results
```python
# Increase candidate pool
results = search("query", n=50)  # Get more
```

### If results not relevant
```python
# Try concept-based search
psql -c "SELECT d.title FROM documents d JOIN passages p ON d.doc_id = p.doc_id JOIN passage_concepts pc ON p.passage_id = pc.passage_id WHERE pc.concept_name = 'specific_concept' LIMIT 10;"
```

---

## POST-HACKATHON

### Save insights
```bash
# Log what queries were useful
echo "Query: X, Found: Y, Used for: Z" >> /home/user/polymath-v4/data/hackathon_insights.log
```

### Harvest new papers found
```bash
python scripts/discover_papers.py "new_method_discovered" --auto-ingest
```

### Update skills with learnings
Edit `/home/user/polymath-v4/skills/hackathon-orchestrator.md` with new patterns.
