#!/usr/bin/env python3
"""
Skill Extraction for Polymath v4

Uses Gemini real-time API to extract actionable skills from passages.

Usage:
    python scripts/extract_skills.py --limit 100
    python scripts/extract_skills.py --doc-id <uuid>
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SKILL_PROMPT = """Analyze this scientific passage and extract any actionable skills or procedures.

A skill is a reusable technique that can be applied to solve problems. Look for:
1. Methods with specific steps
2. Algorithms or protocols
3. Best practices with rationale
4. Troubleshooting procedures

For each skill found, provide:
- name: Short descriptive name (lowercase, hyphenated)
- description: One-sentence summary
- steps: List of concrete steps (if applicable)
- prerequisites: Required knowledge/tools
- skill_type: One of: method, workflow, analysis, integration
- confidence: 0.0-1.0

Return JSON array of skills. If no skills found, return [].

PASSAGE:
{passage}

SOURCE: {title}
"""


def get_db_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def get_skill_candidates(conn, limit: int = 100, doc_id: str = None) -> List[Dict]:
    """Get passages likely to contain skills."""
    cur = conn.cursor()

    # Look for passages with method-related concepts
    if doc_id:
        cur.execute("""
            SELECT DISTINCT p.passage_id, p.passage_text, d.title, d.doc_id
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
            WHERE d.doc_id = %s
            AND LENGTH(p.passage_text) > 200
            AND p.is_superseded = FALSE
            AND NOT EXISTS (
                SELECT 1 FROM paper_skills ps
                WHERE %s = ANY(ps.source_passage_ids)
            )
            ORDER BY pc.confidence DESC NULLS LAST
            LIMIT %s
        """, (doc_id, doc_id, limit))
    else:
        cur.execute("""
            SELECT DISTINCT p.passage_id, p.passage_text, d.title, d.doc_id
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            JOIN passage_concepts pc ON p.passage_id = pc.passage_id
            WHERE pc.concept_type = 'method'
            AND pc.confidence > 0.7
            AND LENGTH(p.passage_text) > 300
            AND p.is_superseded = FALSE
            ORDER BY pc.confidence DESC
            LIMIT %s
        """, (limit,))

    return [
        {
            'passage_id': str(row[0]),
            'passage_text': row[1],
            'title': row[2],
            'doc_id': str(row[3])
        }
        for row in cur.fetchall()
    ]


def extract_skills_from_passage(passage: Dict) -> List[Dict]:
    """Extract skills from a single passage using Gemini."""
    try:
        import google.generativeai as genai

        genai.configure()
        model = genai.GenerativeModel(config.GEMINI_REALTIME_MODEL)

        prompt = SKILL_PROMPT.format(
            passage=passage['passage_text'][:3000],
            title=passage['title']
        )

        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.2,
                'max_output_tokens': 2048,
                'response_mime_type': 'application/json'
            }
        )

        text = response.text.strip()
        # Handle markdown code blocks
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0]

        skills = json.loads(text)
        if isinstance(skills, dict):
            skills = [skills]

        return skills

    except Exception as e:
        logger.warning(f"Error extracting skills: {e}")
        return []


def store_skill(conn, skill: Dict, passage: Dict) -> Optional[str]:
    """Store extracted skill in database."""
    cur = conn.cursor()

    skill_name = skill.get('name', 'unnamed-skill')
    description = skill.get('description', '')
    steps = skill.get('steps', [])
    prerequisites = skill.get('prerequisites', [])
    skill_type = skill.get('skill_type', 'method')

    # Ensure steps and prerequisites are lists
    if isinstance(steps, str):
        steps = [steps]
    if isinstance(prerequisites, str):
        prerequisites = [prerequisites]

    try:
        # Check if skill already exists
        cur.execute("SELECT skill_id, source_passage_ids, source_doc_ids FROM paper_skills WHERE skill_name = %s", (skill_name,))
        existing = cur.fetchone()

        if existing:
            # Merge evidence into existing skill
            skill_id, existing_passages, existing_docs = existing
            existing_passages = existing_passages or []
            existing_docs = existing_docs or []

            # Add new passage/doc if not already present
            new_passage_id = passage['passage_id']
            new_doc_id = passage['doc_id']

            updated_passages = list(set(existing_passages + [new_passage_id]))
            updated_docs = list(set(existing_docs + [new_doc_id]))

            cur.execute("""
                UPDATE paper_skills SET
                    source_passage_ids = %s,
                    source_doc_ids = %s,
                    evidence_count = %s,
                    updated_at = NOW()
                WHERE skill_id = %s
                RETURNING skill_id
            """, (
                updated_passages,
                updated_docs,
                len(updated_passages),
                skill_id
            ))
        else:
            # Create new skill
            cur.execute("""
                INSERT INTO paper_skills (
                    skill_name,
                    skill_type,
                    description,
                    steps,
                    prerequisites,
                    source_passage_ids,
                    source_doc_ids,
                    evidence_count,
                    status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, 1, 'draft')
                RETURNING skill_id
            """, (
                skill_name,
                skill_type,
                description,
                steps,
                prerequisites,
                [passage['passage_id']],
                [passage['doc_id']]
            ))

        result = cur.fetchone()
        conn.commit()
        return str(result[0]) if result else None

    except Exception as e:
        conn.rollback()
        logger.warning(f"Error storing skill: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Extract skills from passages')
    parser.add_argument('--limit', type=int, default=100, help='Max passages to process')
    parser.add_argument('--doc-id', help='Process specific document')
    parser.add_argument('--dry-run', action='store_true', help='Show candidates without extracting')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    conn = get_db_connection()

    # Get candidates
    candidates = get_skill_candidates(conn, args.limit, args.doc_id)
    print(f"Found {len(candidates)} skill candidates")

    if args.dry_run:
        for c in candidates[:10]:
            print(f"\n--- {c['title'][:60]} ---")
            print(c['passage_text'][:200] + "...")
        return

    # Extract skills
    skills_found = 0
    skills_stored = 0

    for i, passage in enumerate(candidates):
        logger.info(f"Processing {i+1}/{len(candidates)}: {passage['title'][:50]}...")

        skills = extract_skills_from_passage(passage)
        skills_found += len(skills)

        for skill in skills:
            if skill.get('confidence', 0) < 0.5:
                continue

            skill_id = store_skill(conn, skill, passage)
            if skill_id:
                skills_stored += 1
                logger.info(f"  Stored: {skill.get('name')}")

    print(f"\nExtracted {skills_found} skills, stored {skills_stored}")

    # Show summary
    cur = conn.cursor()
    cur.execute("""
        SELECT skill_name, skill_type, evidence_count, status
        FROM paper_skills
        ORDER BY created_at DESC
        LIMIT 10
    """)

    print(f"\n{'Recent Skills':<40} {'Type':<15} {'Evidence':<10} {'Status'}")
    print("-" * 80)
    for row in cur.fetchall():
        name, skill_type, evidence, status = row
        print(f"{name[:40]:<40} {(skill_type or 'unknown')[:15]:<15} {evidence or 0:<10} {status}")

    conn.close()


if __name__ == '__main__':
    main()
