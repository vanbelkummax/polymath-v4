#!/usr/bin/env python3
"""
Paper Discovery via CORE API

Searches 130M+ open access papers and ingests them into Polymath.

Features:
- Search by topic, author, year range
- Deduplication against existing documents (DOI, title_hash)
- Direct text ingestion (no PDF download needed - uses CORE's fullText)
- Optional PDF download for archival
- Rate limiting (10 req/min free tier)

Usage:
    python scripts/discover_papers.py "spatial transcriptomics" --limit 50
    python scripts/discover_papers.py "Visium cell segmentation" --year-min 2020 --auto-ingest
    python scripts/discover_papers.py --query "image to gene expression" --limit 100 --download-pdfs
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

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
# Configuration
# ============================================================================

CORE_API_BASE = "https://api.core.ac.uk/v3"
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")

# Rate limiting: 10 requests per minute for free tier
RATE_LIMIT_DELAY = 6.5  # seconds between requests

# Load from .env if not in environment
if not CORE_API_KEY:
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("CORE_API_KEY="):
                    CORE_API_KEY = line.split("=", 1)[1].strip()
                    break


# ============================================================================
# CORE API Client
# ============================================================================

class COREClient:
    """Client for CORE API v3."""

    def __init__(self, api_key: str = CORE_API_KEY):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        })
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def search_works(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        year_min: int = None,
        year_max: int = None,
        full_text_required: bool = True
    ) -> Dict:
        """
        Search for papers/works.

        Args:
            query: Search query (supports title:"X", fullText:"X", etc.)
            limit: Max results (max 100 per request)
            offset: Pagination offset
            year_min: Minimum publication year
            year_max: Maximum publication year
            full_text_required: Only return papers with fullText available

        Returns:
            Dict with 'totalHits', 'results', 'limit', 'offset'
        """
        self._rate_limit()

        # Build query with filters
        q_parts = [query]
        if year_min:
            q_parts.append(f"yearPublished>={year_min}")
        if year_max:
            q_parts.append(f"yearPublished<={year_max}")

        full_query = " AND ".join(q_parts)

        # CORE API needs trailing slash
        url = f"{CORE_API_BASE}/search/works/"
        params = {
            "q": full_query,
            "limit": min(limit, 100),
            "offset": offset
        }

        logger.info(f"CORE API: searching '{full_query[:60]}...' (offset={offset})")

        try:
            resp = self.session.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()

            # Filter for fullText if required
            if full_text_required:
                data['results'] = [r for r in data['results'] if r.get('fullText')]

            logger.info(f"CORE API: {data.get('totalHits', 0)} total hits, {len(data.get('results', []))} returned")
            return data

        except requests.RequestException as e:
            logger.error(f"CORE API error: {e}")
            return {"totalHits": 0, "results": [], "error": str(e)}

    def search_all(
        self,
        query: str,
        max_results: int = 100,
        **kwargs
    ) -> List[Dict]:
        """
        Search with pagination to get more than 100 results.

        Args:
            query: Search query
            max_results: Maximum total results to fetch
            **kwargs: Additional args for search_works

        Returns:
            List of all results
        """
        all_results = []
        offset = 0

        while len(all_results) < max_results:
            batch_limit = min(100, max_results - len(all_results))
            data = self.search_works(query, limit=batch_limit, offset=offset, **kwargs)

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            offset += len(results)

            # Check if we've exhausted results
            if len(results) < batch_limit or offset >= data.get("totalHits", 0):
                break

        return all_results[:max_results]

    def download_pdf(self, download_url: str, output_path: Path) -> bool:
        """Download PDF to local path."""
        self._rate_limit()

        try:
            resp = self.session.get(download_url, timeout=120, stream=True)
            resp.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False


# ============================================================================
# Database Operations
# ============================================================================

def get_db_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def get_title_hash(title: str) -> str:
    """Generate hash for title deduplication."""
    import re
    normalized = re.sub(r'[^a-z0-9]', '', title.lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def get_doc_id(title: str, doi: str = None) -> uuid.UUID:
    """Generate deterministic doc_id."""
    if doi:
        seed = f"doi:{doi}"
    else:
        seed = f"title:{get_title_hash(title)}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def check_existing(conn, doi: str = None, title_hash: str = None) -> Optional[Tuple[str, str]]:
    """
    Check if document already exists.

    Returns:
        Tuple of (doc_id, title) if exists, None otherwise
    """
    cur = conn.cursor()

    if doi:
        cur.execute("SELECT doc_id, title FROM documents WHERE doi = %s", (doi,))
        result = cur.fetchone()
        if result:
            return result

    if title_hash:
        cur.execute("SELECT doc_id, title FROM documents WHERE title_hash = %s", (title_hash,))
        result = cur.fetchone()
        if result:
            return result

    return None


def get_existing_dois(conn) -> set:
    """Get all existing DOIs for fast deduplication."""
    cur = conn.cursor()
    cur.execute("SELECT doi FROM documents WHERE doi IS NOT NULL")
    return {row[0] for row in cur.fetchall()}


def get_existing_title_hashes(conn) -> set:
    """Get all existing title hashes for fast deduplication."""
    cur = conn.cursor()
    cur.execute("SELECT title_hash FROM documents WHERE title_hash IS NOT NULL")
    return {row[0] for row in cur.fetchall()}


# ============================================================================
# Ingestion from CORE fullText
# ============================================================================

# Lazy load embedder
import threading
_embedder = None
_embedder_lock = threading.Lock()


def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                from lib.embeddings.bge_m3 import BGEM3Embedder
                _embedder = BGEM3Embedder()
                _ = _embedder.model
    return _embedder


def ingest_from_core(
    paper: Dict,
    conn,
    compute_embeddings: bool = True,
    batch_name: str = None
) -> Dict:
    """
    Ingest a paper directly from CORE API response (using fullText).

    Args:
        paper: CORE API result dict
        conn: Database connection
        compute_embeddings: Whether to compute embeddings
        batch_name: Batch identifier

    Returns:
        Dict with status, doc_id, passages, etc.
    """
    from lib.ingest.chunking import chunk_text
    from lib.ingest.asset_detector import AssetDetector

    result = {
        'title': paper.get('title', 'Unknown'),
        'doi': paper.get('doi'),
        'status': 'pending',
        'doc_id': None,
        'passages': 0,
        'error': None
    }

    try:
        # Extract metadata
        title = paper.get('title', '').strip()
        if not title:
            result['status'] = 'error'
            result['error'] = 'No title'
            return result

        doi = paper.get('doi')
        arxiv_id = paper.get('arxivId')
        pmid = paper.get('pubmedId')
        year = paper.get('yearPublished')
        abstract = paper.get('abstract', '')
        full_text = paper.get('fullText', '')

        if not full_text:
            result['status'] = 'error'
            result['error'] = 'No fullText'
            return result

        # Extract authors
        authors = [a.get('name', '') for a in paper.get('authors', [])]
        authors_array = authors if authors else None

        # Generate IDs
        title_hash = get_title_hash(title)
        doc_id = get_doc_id(title, doi)
        result['doc_id'] = str(doc_id)

        # Chunk text
        chunks = chunk_text(full_text)
        if not chunks:
            result['status'] = 'error'
            result['error'] = 'No chunks generated'
            return result

        # Compute embeddings
        embeddings = None
        if compute_embeddings:
            logger.info(f"Computing embeddings for {len(chunks)} chunks...")
            embedder = get_embedder()
            texts = [c['content'] for c in chunks]
            embeddings = embedder.encode(texts)

        # Store in database
        cur = conn.cursor()

        # Upsert document
        cur.execute("""
            INSERT INTO documents (
                doc_id, title, title_hash, doi, arxiv_id, pmid,
                authors, year, abstract, source_method, ingest_batch
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                doi = COALESCE(EXCLUDED.doi, documents.doi),
                arxiv_id = COALESCE(EXCLUDED.arxiv_id, documents.arxiv_id),
                pmid = COALESCE(EXCLUDED.pmid, documents.pmid),
                authors = COALESCE(EXCLUDED.authors, documents.authors),
                year = COALESCE(EXCLUDED.year, documents.year),
                abstract = COALESCE(EXCLUDED.abstract, documents.abstract),
                source_method = EXCLUDED.source_method,
                updated_at = NOW()
            RETURNING doc_id
        """, (
            str(doc_id), title, title_hash, doi, arxiv_id, pmid,
            authors_array, year, abstract or None, 'core_api', batch_name
        ))

        # Delete old passages
        cur.execute("DELETE FROM passages WHERE doc_id = %s", (str(doc_id),))

        # Insert passages
        for i, chunk in enumerate(chunks):
            passage_id = uuid.uuid5(doc_id, f"passage:{i}")
            embedding = embeddings[i].tolist() if embeddings is not None else None

            cur.execute("""
                INSERT INTO passages (passage_id, doc_id, passage_text, section, passage_index, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                str(passage_id),
                str(doc_id),
                chunk['content'],
                chunk.get('header'),
                i,
                embedding
            ))

        conn.commit()
        result['passages'] = len(chunks)

        # Asset detection
        detector = AssetDetector(conn)
        passages = [{'passage_id': str(uuid.uuid5(doc_id, f"passage:{i}")),
                    'passage_text': c['content']} for i, c in enumerate(chunks)]
        asset_counts = detector.detect_and_store(str(doc_id), passages)
        result['assets'] = asset_counts

        result['status'] = 'success'
        logger.info(f"Ingested: {title[:50]}... ({len(chunks)} passages)")

    except Exception as e:
        logger.error(f"Error ingesting: {e}")
        result['status'] = 'error'
        result['error'] = str(e)
        try:
            conn.rollback()
        except:
            pass

    return result


# ============================================================================
# Main Discovery Logic
# ============================================================================

def discover_papers(
    query: str,
    max_results: int = 50,
    year_min: int = None,
    year_max: int = None,
    auto_ingest: bool = False,
    download_pdfs: bool = False,
    pdf_output_dir: Path = None,
    compute_embeddings: bool = True,
    dry_run: bool = False
) -> Dict:
    """
    Discover and optionally ingest papers from CORE API.

    Args:
        query: Search query
        max_results: Maximum papers to discover
        year_min: Minimum publication year
        year_max: Maximum publication year
        auto_ingest: Automatically ingest papers
        download_pdfs: Download PDFs to local directory
        pdf_output_dir: Directory for downloaded PDFs
        compute_embeddings: Whether to compute embeddings during ingest
        dry_run: Just search, don't ingest

    Returns:
        Summary dict with discovered papers and results
    """
    client = COREClient()
    conn = get_db_connection()

    # Get existing identifiers for deduplication
    existing_dois = get_existing_dois(conn)
    existing_hashes = get_existing_title_hashes(conn)

    logger.info(f"Existing corpus: {len(existing_dois)} DOIs, {len(existing_hashes)} title hashes")

    # Search CORE
    papers = client.search_all(
        query,
        max_results=max_results,
        year_min=year_min,
        year_max=year_max,
        full_text_required=True
    )

    logger.info(f"Found {len(papers)} papers with fullText")

    # Deduplicate
    new_papers = []
    duplicates = []

    for paper in papers:
        doi = paper.get('doi')
        title = paper.get('title', '')
        title_hash = get_title_hash(title) if title else None

        if doi and doi in existing_dois:
            duplicates.append({'title': title, 'reason': 'DOI exists'})
            continue

        if title_hash and title_hash in existing_hashes:
            duplicates.append({'title': title, 'reason': 'Title exists'})
            continue

        new_papers.append(paper)

    logger.info(f"After deduplication: {len(new_papers)} new papers ({len(duplicates)} duplicates)")

    # Prepare results
    batch_name = f"core_discovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = {
        'query': query,
        'batch_name': batch_name,
        'total_found': len(papers),
        'new_papers': len(new_papers),
        'duplicates': len(duplicates),
        'ingested': 0,
        'failed': 0,
        'papers': [],
        'manual_retrieval_needed': []
    }

    if dry_run:
        results['papers'] = [
            {
                'title': p.get('title'),
                'doi': p.get('doi'),
                'year': p.get('yearPublished'),
                'has_fulltext': bool(p.get('fullText')),
                'has_pdf': bool(p.get('downloadUrl'))
            }
            for p in new_papers
        ]
        return results

    # Process new papers
    if auto_ingest and new_papers:
        # Pre-load embedder
        if compute_embeddings:
            get_embedder()

        for paper in new_papers:
            paper_info = {
                'title': paper.get('title'),
                'doi': paper.get('doi'),
                'year': paper.get('yearPublished'),
                'download_url': paper.get('downloadUrl')
            }

            # Ingest from fullText
            ingest_result = ingest_from_core(
                paper,
                conn,
                compute_embeddings=compute_embeddings,
                batch_name=batch_name
            )

            paper_info['status'] = ingest_result['status']
            paper_info['passages'] = ingest_result.get('passages', 0)
            paper_info['error'] = ingest_result.get('error')

            if ingest_result['status'] == 'success':
                results['ingested'] += 1
            else:
                results['failed'] += 1
                # Track papers that need manual retrieval
                if not paper.get('fullText'):
                    results['manual_retrieval_needed'].append({
                        'title': paper.get('title'),
                        'doi': paper.get('doi'),
                        'download_url': paper.get('downloadUrl')
                    })

            results['papers'].append(paper_info)

            # Download PDF if requested
            if download_pdfs and paper.get('downloadUrl'):
                if pdf_output_dir is None:
                    pdf_output_dir = Path("/home/user/work/polymax/ingest_staging")
                pdf_output_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename
                safe_title = "".join(c for c in paper.get('title', 'paper')[:50] if c.isalnum() or c in ' -_')
                pdf_path = pdf_output_dir / f"{safe_title}.pdf"

                client.download_pdf(paper['downloadUrl'], pdf_path)

    conn.close()
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Discover papers from CORE API (130M+ open access papers)'
    )
    parser.add_argument('query', nargs='?', help='Search query')
    parser.add_argument('--query', '-q', dest='query_opt', help='Search query (alternative)')
    parser.add_argument('--limit', '-l', type=int, default=50, help='Max papers to fetch')
    parser.add_argument('--year-min', type=int, help='Minimum publication year')
    parser.add_argument('--year-max', type=int, help='Maximum publication year')
    parser.add_argument('--auto-ingest', '-a', action='store_true', help='Automatically ingest papers')
    parser.add_argument('--download-pdfs', action='store_true', help='Download PDFs')
    parser.add_argument('--pdf-dir', type=Path, help='PDF output directory')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embedding computation')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Search only, no ingestion')
    parser.add_argument('--output', '-o', type=Path, help='Output JSON file')
    args = parser.parse_args()

    query = args.query or args.query_opt
    if not query:
        parser.error("Query is required")

    if not CORE_API_KEY:
        logger.error("CORE_API_KEY not found. Add it to .env file.")
        sys.exit(1)

    results = discover_papers(
        query=query,
        max_results=args.limit,
        year_min=args.year_min,
        year_max=args.year_max,
        auto_ingest=args.auto_ingest,
        download_pdfs=args.download_pdfs,
        pdf_output_dir=args.pdf_dir,
        compute_embeddings=not args.no_embeddings,
        dry_run=args.dry_run
    )

    # Output
    print(f"\n{'='*60}")
    print(f"DISCOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"Query: {results['query']}")
    print(f"Total found: {results['total_found']}")
    print(f"New papers: {results['new_papers']}")
    print(f"Duplicates: {results['duplicates']}")

    if results.get('ingested'):
        print(f"Ingested: {results['ingested']}")
    if results.get('failed'):
        print(f"Failed: {results['failed']}")

    if results.get('manual_retrieval_needed'):
        print(f"\n--- MANUAL RETRIEVAL NEEDED ({len(results['manual_retrieval_needed'])}) ---")
        for p in results['manual_retrieval_needed'][:10]:
            print(f"  • {p['title'][:60]}...")
            if p.get('doi'):
                print(f"    DOI: {p['doi']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Show sample papers if dry run
    if args.dry_run and results.get('papers'):
        print(f"\n--- SAMPLE PAPERS ---")
        for p in results['papers'][:5]:
            print(f"  • {p['title'][:60]}...")
            print(f"    Year: {p.get('year')}, DOI: {p.get('doi') or 'N/A'}")


if __name__ == '__main__':
    main()
