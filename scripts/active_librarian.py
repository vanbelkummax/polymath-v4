#!/usr/bin/env python3
"""
Active Librarian - Auto-discover missing papers

Finds papers that should be in our corpus based on:
1. High citation counts in our research domains
2. Papers frequently cited in our existing corpus (via DOI mentions)
3. Semantic Scholar for influential papers

Usage:
    python scripts/active_librarian.py --topics "spatial transcriptomics,cell segmentation"
    python scripts/active_librarian.py --analyze-gaps --min-mentions 3
    python scripts/active_librarian.py --auto-ingest --limit 20
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import psycopg2
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# API Clients
# ============================================================================

class SemanticScholarClient:
    """Client for Semantic Scholar API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    RATE_LIMIT_DELAY = 1.0  # 100 req/min limit

    def __init__(self):
        self.session = requests.Session()
        self.last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request = time.time()

    def search_papers(
        self,
        query: str,
        limit: int = 100,
        year_min: int = None,
        fields: str = "paperId,title,authors,year,citationCount,abstract,externalIds,openAccessPdf"
    ) -> List[Dict]:
        """Search for papers by query."""
        self._rate_limit()

        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": fields
        }
        if year_min:
            params["year"] = f"{year_min}-"

        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def get_paper_by_doi(self, doi: str) -> Optional[Dict]:
        """Get paper details by DOI."""
        self._rate_limit()

        url = f"{self.BASE_URL}/paper/DOI:{doi}"
        params = {
            "fields": "paperId,title,authors,year,citationCount,abstract,externalIds,openAccessPdf,tldr"
        }

        try:
            resp = self.session.get(url, params=params, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Could not fetch DOI {doi}: {e}")
            return None


class COREClient:
    """Client for CORE API (reuse from discover_papers)."""

    BASE_URL = "https://api.core.ac.uk/v3"
    RATE_LIMIT_DELAY = 6.5

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}"
        })
        self.last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self.last_request = time.time()

    def search(self, query: str, limit: int = 50, year_min: int = None) -> List[Dict]:
        """Search CORE for papers."""
        self._rate_limit()

        q = query
        if year_min:
            q = f"{query} AND yearPublished>={year_min}"

        url = f"{self.BASE_URL}/search/works/"
        params = {"q": q, "limit": min(limit, 100)}

        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"CORE search failed: {e}")
            return []


# ============================================================================
# Database Operations
# ============================================================================

def get_db_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def get_existing_dois(conn) -> Set[str]:
    """Get all existing DOIs."""
    cur = conn.cursor()
    cur.execute("SELECT doi FROM documents WHERE doi IS NOT NULL")
    return {row[0].lower() for row in cur.fetchall()}


def get_existing_titles(conn) -> Set[str]:
    """Get existing title hashes for dedup."""
    cur = conn.cursor()
    cur.execute("SELECT title_hash FROM documents WHERE title_hash IS NOT NULL")
    return {row[0] for row in cur.fetchall()}


def extract_dois_from_passages(conn, min_mentions: int = 2) -> Dict[str, int]:
    """
    Extract DOIs mentioned in passages that aren't in our corpus.

    Returns dict of {doi: mention_count}
    """
    existing_dois = get_existing_dois(conn)

    cur = conn.cursor()

    # Search passages for DOI patterns
    cur.execute("""
        SELECT passage_text FROM passages
        WHERE passage_text ~ '10\\.[0-9]{4,}/'
        LIMIT 50000
    """)

    # DOI pattern
    doi_pattern = re.compile(r'10\.\d{4,}/[^\s\]>)"\']+')

    doi_counts = Counter()

    for (text,) in cur.fetchall():
        for match in doi_pattern.findall(text):
            # Clean DOI
            doi = match.lower().rstrip('.,;:)]')
            if doi not in existing_dois:
                doi_counts[doi] += 1

    # Filter by min mentions
    return {doi: count for doi, count in doi_counts.items() if count >= min_mentions}


def get_top_concepts(conn, limit: int = 20) -> List[str]:
    """Get most common concepts to identify research areas."""
    cur = conn.cursor()
    cur.execute("""
        SELECT concept_name, COUNT(*) as cnt
        FROM passage_concepts
        WHERE concept_type IN ('domain', 'method', 'technique')
        AND confidence > 0.6
        GROUP BY concept_name
        ORDER BY cnt DESC
        LIMIT %s
    """, (limit,))

    return [row[0].replace('_', ' ') for row in cur.fetchall()]


# ============================================================================
# Main Librarian Logic
# ============================================================================

def analyze_corpus_gaps(conn, min_mentions: int = 2) -> Dict:
    """
    Analyze gaps in the corpus.

    Returns:
        Dict with missing DOIs, research areas, recommendations
    """
    logger.info("Analyzing corpus gaps...")

    # Find DOIs mentioned but not in corpus
    missing_dois = extract_dois_from_passages(conn, min_mentions)
    logger.info(f"Found {len(missing_dois)} DOIs mentioned {min_mentions}+ times but not in corpus")

    # Get top concepts (research areas)
    top_concepts = get_top_concepts(conn, 20)
    logger.info(f"Top research areas: {top_concepts[:5]}")

    # Current corpus stats
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    doc_count = cur.fetchone()[0]

    cur.execute("""
        SELECT year, COUNT(*) FROM documents
        WHERE year IS NOT NULL
        GROUP BY year ORDER BY year DESC LIMIT 5
    """)
    year_dist = dict(cur.fetchall())

    return {
        'corpus_size': doc_count,
        'year_distribution': year_dist,
        'top_concepts': top_concepts,
        'missing_dois': dict(sorted(missing_dois.items(), key=lambda x: -x[1])[:50]),
        'total_missing': len(missing_dois)
    }


def discover_influential_papers(
    topics: List[str],
    year_min: int = 2020,
    limit_per_topic: int = 20
) -> List[Dict]:
    """
    Discover influential papers in given topics using Semantic Scholar.
    """
    s2_client = SemanticScholarClient()
    all_papers = []

    for topic in topics:
        logger.info(f"Searching Semantic Scholar for: {topic}")
        papers = s2_client.search_papers(
            topic,
            limit=limit_per_topic,
            year_min=year_min
        )

        # Sort by citation count
        papers.sort(key=lambda x: x.get('citationCount', 0) or 0, reverse=True)
        all_papers.extend(papers[:limit_per_topic])

    # Deduplicate by paperId
    seen = set()
    unique = []
    for p in all_papers:
        if p.get('paperId') not in seen:
            seen.add(p.get('paperId'))
            unique.append(p)

    return unique


def resolve_missing_dois(
    missing_dois: Dict[str, int],
    limit: int = 20
) -> List[Dict]:
    """
    Resolve missing DOIs to get metadata and open access links.
    """
    s2_client = SemanticScholarClient()
    resolved = []

    # Sort by mention count
    sorted_dois = sorted(missing_dois.items(), key=lambda x: -x[1])[:limit]

    for doi, mentions in sorted_dois:
        logger.info(f"Resolving DOI: {doi} (mentioned {mentions}x)")
        paper = s2_client.get_paper_by_doi(doi)

        if paper:
            paper['doi'] = doi
            paper['mention_count'] = mentions
            paper['has_open_access'] = bool(paper.get('openAccessPdf'))
            resolved.append(paper)
        else:
            resolved.append({
                'doi': doi,
                'mention_count': mentions,
                'title': None,
                'resolved': False
            })

    return resolved


def generate_wishlist(
    conn,
    topics: List[str] = None,
    year_min: int = 2020,
    limit: int = 50
) -> Dict:
    """
    Generate a wishlist of papers to acquire.
    """
    existing_dois = get_existing_dois(conn)

    wishlist = {
        'generated_at': datetime.now().isoformat(),
        'influential_papers': [],
        'missing_citations': [],
        'manual_retrieval': []
    }

    # 1. Find influential papers in topics
    if topics:
        logger.info(f"Finding influential papers in: {topics}")
        influential = discover_influential_papers(topics, year_min, limit // 2)

        for paper in influential:
            doi = paper.get('externalIds', {}).get('DOI', '').lower()
            if doi and doi in existing_dois:
                continue

            entry = {
                'title': paper.get('title'),
                'authors': [a.get('name') for a in paper.get('authors', [])[:3]],
                'year': paper.get('year'),
                'citations': paper.get('citationCount', 0),
                'doi': doi or None,
                'open_access_pdf': paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else None,
                'abstract': paper.get('abstract', '')[:300] + '...' if paper.get('abstract') else None
            }

            if entry['open_access_pdf']:
                wishlist['influential_papers'].append(entry)
            else:
                wishlist['manual_retrieval'].append(entry)

    # 2. Resolve missing citations
    missing_dois = extract_dois_from_passages(conn, min_mentions=3)
    if missing_dois:
        logger.info(f"Resolving {min(len(missing_dois), 20)} frequently cited DOIs...")
        resolved = resolve_missing_dois(missing_dois, limit=20)

        for paper in resolved:
            if paper.get('resolved') is False:
                wishlist['manual_retrieval'].append({
                    'doi': paper['doi'],
                    'mentions': paper['mention_count'],
                    'note': 'Could not resolve via Semantic Scholar'
                })
            elif paper.get('openAccessPdf'):
                wishlist['missing_citations'].append({
                    'title': paper.get('title'),
                    'doi': paper.get('doi'),
                    'mentions': paper.get('mention_count'),
                    'citations': paper.get('citationCount', 0),
                    'open_access_pdf': paper['openAccessPdf'].get('url')
                })
            else:
                wishlist['manual_retrieval'].append({
                    'title': paper.get('title'),
                    'doi': paper.get('doi'),
                    'mentions': paper.get('mention_count'),
                    'citations': paper.get('citationCount', 0)
                })

    return wishlist


def auto_ingest_from_wishlist(wishlist: Dict, limit: int = 10):
    """
    Automatically ingest papers with open access PDFs from wishlist.
    """
    from scripts.discover_papers import COREClient, ingest_from_core, get_db_connection

    core_key = os.environ.get("CORE_API_KEY", "")
    if not core_key:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("CORE_API_KEY="):
                        core_key = line.split("=", 1)[1].strip()
                        break

    if not core_key:
        logger.error("CORE_API_KEY not found")
        return

    core_client = COREClient(core_key)
    conn = get_db_connection()
    ingested = 0

    # Try to ingest papers with open access
    for source in ['influential_papers', 'missing_citations']:
        for paper in wishlist.get(source, []):
            if ingested >= limit:
                break

            if not paper.get('open_access_pdf'):
                continue

            title = paper.get('title', '')
            logger.info(f"Attempting to ingest: {title[:50]}...")

            # Search CORE for fullText
            results = core_client.search(f'title:"{title}"', limit=1)

            if results and results[0].get('fullText'):
                result = ingest_from_core(
                    results[0],
                    conn,
                    compute_embeddings=True,
                    batch_name='active_librarian'
                )

                if result['status'] == 'success':
                    ingested += 1
                    logger.info(f"Ingested: {title[:50]}... ({result['passages']} passages)")
                else:
                    logger.warning(f"Failed to ingest: {result.get('error')}")

    conn.close()
    logger.info(f"Auto-ingested {ingested} papers")
    return ingested


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Active Librarian - discover missing papers')

    parser.add_argument('--topics', help='Comma-separated research topics')
    parser.add_argument('--analyze-gaps', action='store_true', help='Analyze corpus gaps')
    parser.add_argument('--generate-wishlist', action='store_true', help='Generate acquisition wishlist')
    parser.add_argument('--auto-ingest', action='store_true', help='Auto-ingest open access papers')
    parser.add_argument('--min-mentions', type=int, default=2, help='Min citation mentions for gap analysis')
    parser.add_argument('--year-min', type=int, default=2020, help='Minimum publication year')
    parser.add_argument('--limit', type=int, default=50, help='Max papers to process')
    parser.add_argument('--output', '-o', type=Path, help='Output JSON file')

    args = parser.parse_args()

    conn = get_db_connection()

    if args.analyze_gaps:
        gaps = analyze_corpus_gaps(conn, args.min_mentions)

        print(f"\n{'='*60}")
        print("CORPUS GAP ANALYSIS")
        print(f"{'='*60}")
        print(f"Corpus size: {gaps['corpus_size']} documents")
        print(f"Year distribution: {gaps['year_distribution']}")
        print(f"\nTop research areas:")
        for concept in gaps['top_concepts'][:10]:
            print(f"  • {concept}")
        print(f"\nMissing DOIs (mentioned but not in corpus): {gaps['total_missing']}")
        print("Top missing:")
        for doi, count in list(gaps['missing_dois'].items())[:10]:
            print(f"  • {doi} ({count}x)")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(gaps, f, indent=2)

    elif args.generate_wishlist or args.auto_ingest:
        topics = args.topics.split(',') if args.topics else None

        if not topics:
            # Use top concepts as topics
            topics = get_top_concepts(conn, 5)
            logger.info(f"Using auto-detected topics: {topics}")

        wishlist = generate_wishlist(conn, topics, args.year_min, args.limit)

        print(f"\n{'='*60}")
        print("PAPER WISHLIST")
        print(f"{'='*60}")
        print(f"Influential papers (open access): {len(wishlist['influential_papers'])}")
        print(f"Missing citations (open access): {len(wishlist['missing_citations'])}")
        print(f"Manual retrieval needed: {len(wishlist['manual_retrieval'])}")

        if wishlist['influential_papers']:
            print("\n--- INFLUENTIAL (Auto-ingestable) ---")
            for p in wishlist['influential_papers'][:5]:
                print(f"  [{p.get('citations', 0)} cites] {p['title'][:60]}...")

        if wishlist['manual_retrieval']:
            print("\n--- MANUAL RETRIEVAL NEEDED ---")
            for p in wishlist['manual_retrieval'][:10]:
                print(f"  • {p.get('title', p.get('doi', 'Unknown'))[:60]}...")
                if p.get('doi'):
                    print(f"    DOI: {p['doi']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(wishlist, f, indent=2)
            print(f"\nWishlist saved to: {args.output}")

        if args.auto_ingest:
            print("\n--- AUTO-INGESTING ---")
            ingested = auto_ingest_from_wishlist(wishlist, args.limit)
            print(f"Auto-ingested: {ingested} papers")

    conn.close()


if __name__ == '__main__':
    main()
