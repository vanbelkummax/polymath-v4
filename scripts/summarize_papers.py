#!/usr/bin/env python3
"""
Multi-Paper Summarization - Literature Review Generator

Generates literature review summaries from multiple papers.

Usage:
    python scripts/summarize_papers.py --query "spatial transcriptomics methods"
    python scripts/summarize_papers.py --query "cell segmentation deep learning" --top-k 10
    python scripts/summarize_papers.py --doc-ids uuid1,uuid2,uuid3
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.search.hybrid_search import HybridSearcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# LLM Clients
# ============================================================================

def get_gemini_response(prompt: str, model: str = None) -> str:
    """Get response from Gemini API."""
    try:
        from google import genai

        client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT,
            location=config.GCP_LOCATION
        )

        model_name = model or config.GEMINI_REALTIME_MODEL

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 4096
            }
        )

        return response.text

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


def get_anthropic_response(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Get response from Anthropic Claude API."""
    try:
        import anthropic

        client = anthropic.Anthropic()

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return None


# ============================================================================
# Summarization Prompts
# ============================================================================

LITERATURE_REVIEW_PROMPT = """You are a scientific writer creating a literature review section.

TOPIC: {topic}

Below are passages from {num_papers} relevant papers. Synthesize these into a coherent literature review that:

1. Identifies major themes and approaches
2. Highlights key findings and contributions
3. Notes methodological advances
4. Identifies gaps and future directions
5. Cites papers using [Author et al., Year] format

PASSAGES:

{passages}

---

Write a well-structured literature review (500-800 words) that synthesizes the above passages. Use academic writing style. Include inline citations.

Begin with a topic sentence summarizing the field's current state.
"""

COMPARISON_PROMPT = """Compare and contrast the following papers on {topic}.

PAPERS:

{paper_summaries}

---

Create a comparison table highlighting:
1. Methods used
2. Key findings
3. Datasets
4. Strengths and limitations

Then write a 200-word synthesis of how these papers relate to each other.
"""


# ============================================================================
# Core Functions
# ============================================================================

def get_db_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def get_passages_by_query(
    conn,
    query: str,
    top_k: int = 20,
    passages_per_paper: int = 3
) -> List[Dict]:
    """
    Get relevant passages via hybrid search, grouped by paper.

    Returns:
        List of dicts with paper info and passages
    """
    searcher = HybridSearcher(rerank=True)
    results = searcher.hybrid_search(query, n=top_k * passages_per_paper)

    # Group by doc_id
    papers = {}
    for r in results:
        if r.doc_id not in papers:
            papers[r.doc_id] = {
                'doc_id': r.doc_id,
                'title': r.title,
                'passages': []
            }
        if len(papers[r.doc_id]['passages']) < passages_per_paper:
            papers[r.doc_id]['passages'].append({
                'text': r.passage_text,
                'score': r.score
            })

    # Get author/year info
    cur = conn.cursor()
    for doc_id in papers:
        cur.execute("""
            SELECT authors, year FROM documents WHERE doc_id = %s
        """, (doc_id,))
        row = cur.fetchone()
        if row:
            authors = row[0] or []
            first_author = authors[0].split(',')[0] if authors else 'Unknown'
            papers[doc_id]['first_author'] = first_author
            papers[doc_id]['year'] = row[1] or 'N/A'
            papers[doc_id]['citation'] = f"[{first_author} et al., {row[1] or 'N/A'}]"

    return list(papers.values())[:top_k]


def get_passages_by_doc_ids(
    conn,
    doc_ids: List[str],
    passages_per_paper: int = 5
) -> List[Dict]:
    """Get passages for specific documents."""
    papers = []
    cur = conn.cursor()

    for doc_id in doc_ids:
        # Get document info
        cur.execute("""
            SELECT title, authors, year FROM documents WHERE doc_id = %s
        """, (doc_id,))
        row = cur.fetchone()
        if not row:
            continue

        title, authors, year = row
        authors = authors or []
        first_author = authors[0].split(',')[0] if authors else 'Unknown'

        # Get top passages
        cur.execute("""
            SELECT passage_text
            FROM passages
            WHERE doc_id = %s
            AND is_superseded = FALSE
            AND LENGTH(passage_text) > 200
            ORDER BY passage_index
            LIMIT %s
        """, (doc_id, passages_per_paper))

        passages = [{'text': r[0], 'score': 1.0} for r in cur.fetchall()]

        papers.append({
            'doc_id': doc_id,
            'title': title,
            'first_author': first_author,
            'year': year or 'N/A',
            'citation': f"[{first_author} et al., {year or 'N/A'}]",
            'passages': passages
        })

    return papers


def format_passages_for_prompt(papers: List[Dict]) -> str:
    """Format passages for LLM prompt."""
    formatted = []

    for paper in papers:
        paper_text = f"\n### {paper['title']}\n"
        paper_text += f"Citation: {paper['citation']}\n\n"

        for i, p in enumerate(paper['passages'], 1):
            paper_text += f"Passage {i}:\n{p['text'][:1500]}\n\n"

        formatted.append(paper_text)

    return "\n---\n".join(formatted)


def generate_literature_review(
    topic: str,
    papers: List[Dict],
    llm: str = 'gemini'
) -> str:
    """Generate literature review from papers."""
    passages_text = format_passages_for_prompt(papers)

    prompt = LITERATURE_REVIEW_PROMPT.format(
        topic=topic,
        num_papers=len(papers),
        passages=passages_text
    )

    logger.info(f"Generating literature review using {llm}...")

    if llm == 'gemini':
        response = get_gemini_response(prompt)
    elif llm == 'anthropic':
        response = get_anthropic_response(prompt)
    else:
        raise ValueError(f"Unknown LLM: {llm}")

    if not response:
        return "Error generating literature review"

    # Add references section
    references = "\n\n## References\n\n"
    for paper in papers:
        references += f"- {paper['citation']} {paper['title']}\n"

    return response + references


def generate_comparison(
    topic: str,
    papers: List[Dict],
    llm: str = 'gemini'
) -> str:
    """Generate comparison of papers."""
    summaries = []
    for paper in papers:
        summary = f"**{paper['title']}** {paper['citation']}\n"
        summary += f"Key passages:\n"
        for p in paper['passages'][:2]:
            summary += f"- {p['text'][:500]}...\n"
        summaries.append(summary)

    prompt = COMPARISON_PROMPT.format(
        topic=topic,
        paper_summaries="\n---\n".join(summaries)
    )

    logger.info(f"Generating comparison using {llm}...")

    if llm == 'gemini':
        response = get_gemini_response(prompt)
    elif llm == 'anthropic':
        response = get_anthropic_response(prompt)
    else:
        raise ValueError(f"Unknown LLM: {llm}")

    return response or "Error generating comparison"


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate literature reviews from papers')

    parser.add_argument('--query', '-q', help='Search query to find relevant papers')
    parser.add_argument('--doc-ids', help='Comma-separated document IDs')
    parser.add_argument('--top-k', type=int, default=10, help='Number of papers to include')
    parser.add_argument('--passages-per-paper', type=int, default=3, help='Passages per paper')
    parser.add_argument('--compare', action='store_true', help='Generate comparison instead of review')
    parser.add_argument('--llm', choices=['gemini', 'anthropic'], default='gemini', help='LLM to use')
    parser.add_argument('--output', '-o', type=Path, help='Output markdown file')
    parser.add_argument('--dry-run', action='store_true', help='Show papers without generating')

    args = parser.parse_args()

    if not args.query and not args.doc_ids:
        parser.error("Either --query or --doc-ids required")

    conn = get_db_connection()

    # Get papers
    if args.query:
        logger.info(f"Searching for papers matching: {args.query}")
        papers = get_passages_by_query(
            conn, args.query,
            top_k=args.top_k,
            passages_per_paper=args.passages_per_paper
        )
    else:
        doc_ids = args.doc_ids.split(',')
        logger.info(f"Getting passages for {len(doc_ids)} documents")
        papers = get_passages_by_doc_ids(
            conn, doc_ids,
            passages_per_paper=args.passages_per_paper
        )

    if not papers:
        logger.error("No papers found")
        return

    logger.info(f"Found {len(papers)} papers")

    # Show paper list
    print(f"\n{'='*60}")
    print(f"PAPERS ({len(papers)})")
    print(f"{'='*60}")
    for p in papers:
        print(f"  {p['citation']} {p['title'][:60]}...")

    if args.dry_run:
        return

    # Generate summary
    topic = args.query or "the selected papers"

    if args.compare:
        output = generate_comparison(topic, papers, args.llm)
    else:
        output = generate_literature_review(topic, papers, args.llm)

    # Display/save
    print(f"\n{'='*60}")
    print("GENERATED REVIEW")
    print(f"{'='*60}")
    print(output)

    if args.output:
        # Add metadata header
        header = f"""---
title: Literature Review - {topic}
generated: {datetime.now().isoformat()}
papers: {len(papers)}
llm: {args.llm}
---

"""
        with open(args.output, 'w') as f:
            f.write(header + output)
        logger.info(f"Saved to {args.output}")

    conn.close()


if __name__ == '__main__':
    main()
