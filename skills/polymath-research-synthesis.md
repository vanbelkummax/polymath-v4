---
name: polymath-research-synthesis
description: Generate comprehensive literature reviews and research syntheses using the Polymath knowledge base combined with external sources (PubMed, bioRxiv, OpenTargets). Use when asked to summarize "what's known about X", write background sections, or synthesize research on a topic.
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob, WebFetch, WebSearch]
---

# Polymath Research Synthesis

## Overview

Synthesize knowledge from the Polymath knowledge base (1,900+ papers, 160K passages) with external sources to generate comprehensive, citation-backed research summaries. This skill combines local semantic search with MCP servers (vanderbilt-professors, chembl, clinical-trials, open-targets) for complete coverage.

## When to Use This Skill

Use this skill when:
- Asked "What's known about X?" or "Summarize research on Y"
- Writing Introduction or Background sections for papers/grants
- Preparing literature reviews on a topic
- Synthesizing evidence across multiple papers
- Answering research questions with citations

## Workflow

### Phase 1: Polymath Search (Local KB)

```python
cd /home/user/polymath-v4

# Search local knowledge base
python -c "
from lib.search.hybrid_search import search

results = search('YOUR_QUERY_HERE', n=20)
for r in results:
    print(f\"## {r['title']} ({r.get('year', 'n.d.')})\")
    print(f\"Score: {r['score']:.3f}\")
    print(f\"Text: {r['text'][:300]}...\")
    print()
"
```

### Phase 2: Concept-Based Expansion

```sql
-- Find related concepts
SELECT concept_name, concept_type, COUNT(*) as freq
FROM passage_concepts
WHERE concept_name ILIKE '%your_term%'
GROUP BY concept_name, concept_type
ORDER BY freq DESC
LIMIT 20;

-- Find passages with specific concepts
SELECT p.passage_text, d.title, d.year
FROM passages p
JOIN documents d ON p.doc_id = d.doc_id
JOIN passage_concepts pc ON p.passage_id = pc.passage_id
WHERE pc.concept_name = 'specific_concept'
AND pc.confidence > 0.7
LIMIT 10;
```

### Phase 3: External Sources (MCP Servers)

Use these MCP tools to supplement local results:

```
# Vanderbilt professors (830 papers from 7 faculty)
mcp__vanderbilt-professors__search_all_professors(query="your topic")

# For drug/compound questions
mcp__chembl__compound_search(name="drug_name")
mcp__chembl__get_mechanism(molecule_chembl_id="CHEMBL...")

# For clinical trial context
mcp__clinical-trials__search_trials(condition="disease", status=["RECRUITING"])

# For target/disease associations
mcp__open-targets__search_entities(query_strings=["gene", "disease"])
```

### Phase 4: Synthesis Structure

Generate output in this format:

```markdown
# Research Synthesis: [TOPIC]

## Executive Summary
[2-3 sentence overview of the field]

## Key Findings

### [Theme 1]
[Synthesized findings with citations]

Evidence:
- "Direct quote from passage" (Author et al., Year)
- "Another supporting quote" (Author et al., Year)

### [Theme 2]
[Continue pattern...]

## Methods Landscape
[What techniques are used in this area]

## Open Questions & Gaps
[What remains unknown, citing papers that mention limitations]

## Key References
1. Author et al. (Year). Title. Journal. [From Polymath]
2. Author et al. (Year). Title. [From MCP: vanderbilt-professors]
```

## Quality Checklist

- [ ] Searched Polymath with multiple query variants
- [ ] Checked concept graph for related terms
- [ ] Queried relevant MCP servers (professors, ChEMBL, trials)
- [ ] Every claim has a citation
- [ ] Identified conflicting findings (if any)
- [ ] Noted gaps and limitations
- [ ] Output is in flowing prose, not bullet points (for final version)

## Example Queries

```bash
# Generate synthesis on spatial transcriptomics methods
python scripts/summarize_papers.py --query "spatial transcriptomics prediction from H&E" --top-k 15

# Find gaps in the literature
python scripts/active_librarian.py --analyze-gaps --min-mentions 3
```

## Integration with Scientific Writing

After generating synthesis, use the `scientific-writing` skill to convert to manuscript-ready prose with proper citations (APA, Nature, Vancouver style).
