---
name: polymath-code-paper-bridge
description: Bridge between code implementations and research papers. Find the paper for a GitHub repo, find implementations for a method, trace code to its theoretical foundations, and connect implementations across papers. Leverages 1,800+ indexed repos linked to 1,900+ papers.
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, WebFetch]
---

# Polymath Code-Paper Bridge

## Overview

Connect code implementations to their research papers and vice versa. The Polymath knowledge base contains 1,800+ GitHub repositories linked to 1,900+ papers, enabling bidirectional discovery:

- **Paper → Code**: "Find implementations of this method"
- **Code → Paper**: "What paper describes this repo?"
- **Method → All**: "Show me all implementations of attention mechanisms"

## When to Use This Skill

Use this skill when:
- Looking for code implementation of a method from a paper
- Finding the paper that describes a GitHub repo
- Understanding the theory behind a codebase
- Comparing implementations across papers
- Replicating a paper's results
- Finding datasets mentioned in papers with available code

## Database Schema

```sql
-- Papers with their repos
SELECT d.title, d.year, pr.repo_url
FROM documents d
JOIN paper_repos pr ON d.doc_id = pr.doc_id
WHERE d.title ILIKE '%your_paper%';

-- Repos with metadata
SELECT r.repo_url, r.name, r.stars, r.language, r.description
FROM repositories r
WHERE r.name ILIKE '%your_repo%';

-- Repo content (README, docstrings)
SELECT rp.passage_text, r.repo_url
FROM repo_passages rp
JOIN repositories r ON rp.repo_id = r.repo_id
WHERE rp.passage_text ILIKE '%your_method%'
LIMIT 10;
```

## Workflows

### 1. Find Code for a Paper Method

```python
cd /home/user/polymath-v4

# Step 1: Find the paper
python -c "
from lib.search.hybrid_search import search
results = search('attention mechanism transformer', n=10)
for r in results:
    print(f'{r[\"title\"]} ({r.get(\"year\", \"n.d.\")})')
    print(f'  Doc ID: {r.get(\"doc_id\", \"N/A\")}')
"

# Step 2: Find linked repos
psql -U polymath -d polymath -c "
SELECT pr.repo_url, d.title
FROM paper_repos pr
JOIN documents d ON pr.doc_id = d.doc_id
WHERE d.title ILIKE '%attention%'
LIMIT 20;
"
```

### 2. Find Paper for a GitHub Repo

```sql
-- Direct lookup
SELECT d.title, d.year, d.authors, pr.repo_url
FROM documents d
JOIN paper_repos pr ON d.doc_id = pr.doc_id
WHERE pr.repo_url ILIKE '%scanpy%';

-- Search repo content then link to paper
SELECT DISTINCT d.title, d.year, r.repo_url
FROM repo_passages rp
JOIN repositories r ON rp.repo_id = r.repo_id
JOIN paper_repos pr ON r.repo_url = pr.repo_url
JOIN documents d ON pr.doc_id = d.doc_id
WHERE rp.passage_text ILIKE '%normalize%counts%'
LIMIT 10;
```

### 3. Find All Implementations of a Method

```python
# Search both papers and repos
python -c "
from lib.search.hybrid_search import search

# Search papers
print('=== Papers ===')
for r in search('graph neural network implementation', n=5):
    print(f'{r[\"title\"]}: {r[\"text\"][:100]}')

# Search repo READMEs
print()
print('=== Repo Code ===')
"

# SQL for repo-specific search
psql -U polymath -d polymath -c "
SELECT r.name, r.repo_url, r.stars, r.language,
       LEFT(rp.passage_text, 200) as readme_excerpt
FROM repo_passages rp
JOIN repositories r ON rp.repo_id = r.repo_id
WHERE rp.passage_text ILIKE '%graph neural network%'
ORDER BY r.stars DESC NULLS LAST
LIMIT 10;
"
```

### 4. Compare Implementations

```sql
-- Find multiple implementations of same method
SELECT r.name, r.repo_url, r.stars, r.language, d.title as paper
FROM repositories r
LEFT JOIN paper_repos pr ON r.repo_url = pr.repo_url
LEFT JOIN documents d ON pr.doc_id = d.doc_id
WHERE r.name ILIKE '%transformer%'
   OR r.description ILIKE '%transformer%'
ORDER BY r.stars DESC NULLS LAST
LIMIT 20;
```

### 5. Trace Theory to Implementation

```python
# Find theoretical foundation
python -c "
from lib.search.hybrid_search import search

# Get theory
print('=== Theoretical Papers ===')
for r in search('variational autoencoder theory derivation', n=5):
    print(f'{r[\"title\"]}')
    print(f'  {r[\"text\"][:150]}...')
    print()
"

# Then find implementations
psql -U polymath -d polymath -c "
SELECT r.name, r.stars, r.language, r.repo_url
FROM repositories r
JOIN repo_passages rp ON r.repo_id = rp.repo_id
WHERE rp.passage_text ILIKE '%variational%autoencoder%'
   OR rp.passage_text ILIKE '%VAE%'
ORDER BY r.stars DESC NULLS LAST
LIMIT 10;
"
```

## Quick Commands

```bash
cd /home/user/polymath-v4

# Find repos for a paper title
psql -U polymath -d polymath -c "
SELECT pr.repo_url FROM paper_repos pr
JOIN documents d ON pr.doc_id = d.doc_id
WHERE d.title ILIKE '%your paper title%';
"

# Find paper for a repo
psql -U polymath -d polymath -c "
SELECT d.title, d.year, d.authors FROM documents d
JOIN paper_repos pr ON d.doc_id = pr.doc_id
WHERE pr.repo_url ILIKE '%repo_name%';
"

# Search repo READMEs
psql -U polymath -d polymath -c "
SELECT r.name, r.repo_url, LEFT(rp.passage_text, 150)
FROM repo_passages rp
JOIN repositories r ON rp.repo_id = r.repo_id
WHERE rp.passage_text ILIKE '%your_method%'
LIMIT 5;
"

# Top starred repos in KB
psql -U polymath -d polymath -c "
SELECT name, stars, language, repo_url
FROM repositories
WHERE stars IS NOT NULL
ORDER BY stars DESC
LIMIT 20;
"
```

## Replication Workflow

When trying to replicate a paper:

1. **Find the paper's repo**
```sql
SELECT pr.repo_url FROM paper_repos pr
JOIN documents d ON pr.doc_id = d.doc_id
WHERE d.title ILIKE '%paper title%';
```

2. **Check repo quality**
```sql
SELECT r.stars, r.language, r.updated_at, r.description
FROM repositories r
WHERE r.repo_url ILIKE '%repo_name%';
```

3. **Find dataset mentions**
```python
search('paper title dataset download', n=10)
```

4. **Cross-reference with other implementations**
```sql
-- Find alternative implementations
SELECT DISTINCT r.name, r.repo_url, r.stars
FROM repositories r
JOIN repo_passages rp ON r.repo_id = rp.repo_id
WHERE rp.passage_text ILIKE '%method_name%'
AND r.repo_url NOT LIKE '%original_repo%'
ORDER BY r.stars DESC NULLS LAST;
```

## Statistics

```sql
-- Current coverage
SELECT
  (SELECT COUNT(*) FROM documents) as papers,
  (SELECT COUNT(*) FROM repositories) as repos,
  (SELECT COUNT(*) FROM paper_repos) as paper_repo_links,
  (SELECT COUNT(*) FROM repo_passages) as repo_passages;
```

## Integration with Other Skills

- **polymath-research-synthesis**: Find papers, then get their code
- **verify_repos.py**: Check if repos are still active and installable
- **scientific-writing**: Cite both paper and code implementation
