---
name: polymath-exam-prep
description: Prepare for qualifying exams (PQE), comprehensive exams, and oral defenses using the Polymath knowledge base. Generates practice questions, identifies knowledge gaps, creates study materials, and provides evidence-backed answers. Designed for MD-PhD and graduate student exam preparation.
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob]
---

# Polymath Exam Prep

## Overview

Leverage the Polymath knowledge base to prepare for qualifying exams, comprehensive exams, and oral defenses. This skill generates practice questions at different difficulty levels, identifies knowledge gaps, and provides evidence-backed answers with citations.

## When to Use This Skill

Use this skill when:
- Preparing for PhD qualifying exams (PQE)
- Studying for comprehensive/preliminary exams
- Practicing for oral defense questions
- Reviewing a field systematically
- Testing depth of understanding on a topic
- Identifying gaps in your knowledge

## Exam Prep Modes

### Mode 1: Question Generation

Generate exam-style questions from the knowledge base:

```python
cd /home/user/polymath-v4

python -c "
from lib.search.hybrid_search import search

# Search for a topic
topic = 'spatial transcriptomics'
results = search(topic, n=30)

# Extract key concepts for question generation
concepts = set()
for r in results:
    # Get text for question generation
    print(f'Paper: {r[\"title\"]}')
    print(f'Key passage: {r[\"text\"][:200]}...')
    print()
"
```

**Question Templates by Difficulty:**

**Foundational (Define/Describe):**
- What is [CONCEPT] and why is it important?
- Describe the key steps in [METHOD].
- Compare and contrast [A] vs [B].

**Integrative (Analyze/Synthesize):**
- How does [METHOD A] address limitations of [METHOD B]?
- What are the trade-offs between [APPROACH 1] and [APPROACH 2]?
- Explain how [FINDING] changes our understanding of [FIELD].

**Critical (Evaluate/Design):**
- Critique the experimental design of [STUDY].
- Design an experiment to test [HYPOTHESIS].
- What are the limitations of current approaches to [PROBLEM]?

### Mode 2: Knowledge Gap Analysis

```sql
-- Find topics you should know (frequently mentioned)
SELECT concept_name, COUNT(*) as mentions
FROM passage_concepts
WHERE concept_type IN ('methods', 'problems')
GROUP BY concept_name
HAVING COUNT(*) > 5
ORDER BY mentions DESC
LIMIT 50;

-- Compare your reading list to the KB
-- (identify papers you haven't read)
```

### Mode 3: Evidence-Backed Answers

When answering a question, use this template:

```markdown
## Question: [EXAM QUESTION]

### Answer

[Your synthesized answer in 2-3 paragraphs]

### Supporting Evidence

1. **[Key Point 1]**
   - "Direct quote from paper" (Author et al., Year)
   - Source: [Paper title]

2. **[Key Point 2]**
   - "Supporting evidence" (Author et al., Year)
   - Source: [Paper title]

### Related Concepts to Know
- [Concept 1]: Brief definition
- [Concept 2]: Brief definition

### Potential Follow-up Questions
- [Question examiner might ask next]
- [Another follow-up]
```

### Mode 4: Topic Deep Dive

For comprehensive review of a topic:

```bash
# Get all papers on a topic
python -c "
from lib.search.hybrid_search import search
results = search('YOUR_TOPIC', n=50)
for i, r in enumerate(results[:20], 1):
    print(f'{i}. {r[\"title\"]} ({r.get(\"year\", \"n.d.\")})')
    print(f'   Key: {r[\"text\"][:150]}...')
    print()
"

# Find methodological details
python -c "
from lib.search.hybrid_search import search
results = search('YOUR_TOPIC methods experimental design', n=20)
for r in results:
    print(f'Method: {r[\"text\"][:200]}')
    print()
"
```

## Study Session Workflow

### Before Session
1. Pick a topic/theme
2. Generate 5-10 practice questions
3. Set timer (simulate exam conditions)

### During Session
```bash
# For each question:
# 1. Attempt answer from memory
# 2. Then search KB for evidence
python -c "
from lib.search.hybrid_search import search
results = search('question keywords', n=10)
for r in results:
    print(r['title'], r['text'][:200])
"
```

### After Session
- Note gaps in your knowledge
- Add missing papers to reading list
- Create flashcards for weak areas

## Exam-Specific Preparation

### Qualifying Exam (PQE)
Focus areas:
- Specific Aims defense
- Methodology justification
- Alternative approaches
- Preliminary data interpretation
- Broader impacts

```python
# Find papers that support your specific aims
search("YOUR_AIM_KEYWORDS", n=20)

# Find potential criticisms/alternatives
search("limitations of YOUR_METHOD", n=10)
```

### Comprehensive Exam
Focus areas:
- Breadth across your field
- Historical context (seminal papers)
- Current state of the art
- Future directions

```sql
-- Find seminal papers (older, highly connected)
SELECT d.title, d.year, d.authors
FROM documents d
WHERE d.year < 2015
ORDER BY d.year ASC
LIMIT 20;
```

### Oral Defense
Focus areas:
- Your thesis contributions
- Limitations and future work
- Broader significance

## Quick Reference Commands

```bash
cd /home/user/polymath-v4

# Quick topic search
python -c "from lib.search.hybrid_search import search; [print(f'{r[\"title\"]}: {r[\"text\"][:100]}') for r in search('TOPIC', n=5)]"

# Find definitions
python -c "from lib.search.hybrid_search import search; [print(r['text'][:300]) for r in search('what is CONCEPT definition', n=3)]"

# Find methods
python -c "from lib.search.hybrid_search import search; [print(r['text'][:300]) for r in search('TOPIC methods protocol', n=5)]"

# Find limitations/gaps
python -c "from lib.search.hybrid_search import search; [print(r['text'][:300]) for r in search('TOPIC limitations challenges future', n=5)]"
```

## Integration with Other Skills

- **hypothesis-generation**: Generate novel research questions
- **scientific-writing**: Convert answers to manuscript prose
- **polymath-research-synthesis**: Create comprehensive topic reviews
