#!/usr/bin/env python3
"""
PDF Ingestion for Polymath v4

Ingests PDFs with:
- Text extraction (PyMuPDF)
- Chunking (by headers or fixed size)
- BGE-M3 embeddings (1024-dim)
- Asset detection (GitHub, HF, DOI)

Usage:
    python scripts/ingest_pdf.py paper.pdf
    python scripts/ingest_pdf.py /path/to/papers/*.pdf --workers 4
"""

import argparse
import logging
import sys
import time
import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config
from lib.ingest.pdf_parser import PDFParser
from lib.ingest.chunking import chunk_text
from lib.ingest.asset_detector import AssetDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy load embedder (GPU resource) - thread-safe via module-level lock
import threading
_embedder = None
_embedder_lock = threading.Lock()

def get_embedder():
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:  # Double-check after acquiring lock
                from lib.embeddings.bge_m3 import BGEM3Embedder
                _embedder = BGEM3Embedder()
                # Force model loading now (before threads use it)
                _ = _embedder.model
    return _embedder


def get_db_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def get_title_hash(title: str) -> str:
    """Generate hash for title deduplication."""
    import re
    normalized = re.sub(r'[^a-z0-9]', '', title.lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def extract_title_from_text(text: str, fallback: str) -> str:
    """
    Extract title from PDF text with smart heuristics.

    Skips:
    - Short lines (< 10 chars)
    - Lines that are mostly punctuation
    - URL lines
    - Header/footer patterns
    """
    import re

    lines = text.strip().split('\n')

    for line in lines[:20]:  # Check first 20 lines
        line = line.strip()

        # Skip short lines
        if len(line) < 10:
            continue

        # Skip lines that are mostly punctuation/whitespace
        alpha_chars = sum(1 for c in line if c.isalnum())
        if alpha_chars < len(line) * 0.5:
            continue

        # Skip URLs and lines containing URLs
        if 'http' in line.lower() or 'doi.org' in line.lower() or 'dl.acm.org' in line.lower():
            continue

        # Skip lines starting with common non-title patterns
        if line.lower().startswith(('latest update', 'pdf download', 'open access')):
            continue

        # Skip common header patterns
        skip_patterns = [
            r'^page\s*\d+',
            r'^\d+\s*$',
            r'^(abstract|introduction|references|acknowledgments)',
            r'^figure\s*\d+',
            r'^table\s*\d+',
            r'^(poster|article|review|letter|short\s*paper)',  # document type markers
        ]
        if any(re.match(p, line.lower()) for p in skip_patterns):
            continue

        # Found a good title candidate
        return line[:200]

    # Fallback to filename
    return fallback


def extract_doi_from_text(text: str) -> Optional[str]:
    """Extract DOI from PDF text if present."""
    import re
    # Match DOI patterns like 10.1234/something
    patterns = [
        r'doi[:\s]*(\d{2}\.\d{4,}/[^\s]+)',
        r'(10\.\d{4,}/[^\s]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            doi = match.group(1).rstrip('.,)')
            return doi
    return None


def get_pdf_hash(pdf_path: Path) -> str:
    """Generate hash of PDF content for duplicate detection."""
    with open(pdf_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def load_zotero_metadata(csv_path: str) -> dict:
    """
    Load Zotero CSV into a lookup dict keyed by Linux PDF path.

    Returns dict: {pdf_path: {title, doi, authors, year, abstract, venue, zotero_key}}
    """
    import csv

    metadata = {}

    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_path = row.get('_pdf_path_linux', '').strip()
            if not pdf_path:
                continue

            metadata[pdf_path] = {
                'title': row.get('Title', '').strip(),
                'doi': row.get('DOI', '').strip(),
                'authors': row.get('Author', '').strip(),
                'year': row.get('Publication Year', '').strip(),
                'abstract': row.get('Abstract Note', '').strip(),
                'venue': row.get('Publication Title', '').strip(),
                'zotero_key': row.get('Key', '').strip(),
            }

    logger.info(f"Loaded Zotero metadata for {len(metadata)} PDFs")
    return metadata


# Global Zotero metadata cache
_zotero_metadata = None


def get_zotero_metadata(pdf_path: str) -> Optional[dict]:
    """Get Zotero metadata for a PDF path if available."""
    global _zotero_metadata
    if _zotero_metadata is None:
        return None
    return _zotero_metadata.get(str(pdf_path))


def get_doc_id(title: str, doi: str = None) -> uuid.UUID:
    """Generate deterministic doc_id."""
    if doi:
        seed = f"doi:{doi}"
    else:
        seed = f"title:{get_title_hash(title)}"
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def ingest_single_pdf(
    pdf_path: Path,
    conn=None,
    compute_embeddings: bool = True,
    detect_assets: bool = True,
    batch_name: str = None
) -> dict:
    """
    Ingest a single PDF.

    Returns:
        dict with doc_id, title, passages, status
    """
    close_conn = False
    if conn is None:
        conn = get_db_connection()
        close_conn = True

    result = {
        'path': str(pdf_path),
        'doc_id': None,
        'title': None,
        'passages': 0,
        'status': 'pending',
        'error': None
    }

    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            result['status'] = 'error'
            result['error'] = 'File not found'
            return result

        # Parse PDF
        logger.info(f"Parsing: {pdf_path.name}")
        parser = PDFParser()
        parse_result = parser.parse(pdf_path)

        if not parse_result.has_text:
            result['status'] = 'error'
            result['error'] = 'No text extracted'
            return result

        # Try Zotero metadata first (richest source)
        zotero = get_zotero_metadata(str(pdf_path))

        if zotero and zotero.get('title'):
            # Use Zotero metadata
            title = zotero['title']
            doi = zotero.get('doi') or extract_doi_from_text(parse_result.text)
            authors = zotero.get('authors', '')
            year = zotero.get('year', '')
            abstract = zotero.get('abstract', '')
            venue = zotero.get('venue', '')
            zotero_key = zotero.get('zotero_key', '')
            metadata_source = 'zotero'
        else:
            # Fall back to PDF extraction
            title = extract_title_from_text(parse_result.text, pdf_path.stem)
            doi = extract_doi_from_text(parse_result.text)
            authors = ''
            year = ''
            abstract = ''
            venue = ''
            zotero_key = ''
            metadata_source = 'pdf_extraction'

        # Generate PDF hash for duplicate detection
        pdf_hash = get_pdf_hash(pdf_path)

        # Generate doc_id (use DOI if available for stability)
        doc_id = get_doc_id(title, doi)
        result['doc_id'] = str(doc_id)
        result['title'] = title
        result['doi'] = doi
        result['metadata_source'] = metadata_source

        # Check for existing document with same DOI (skip if exists)
        cur = conn.cursor()
        if doi:
            cur.execute("SELECT doc_id, title FROM documents WHERE doi = %s", (doi,))
            existing = cur.fetchone()
            if existing:
                result['status'] = 'skipped'
                result['error'] = f'DOI already exists: {existing[1][:50]}...'
                logger.info(f"Skipped (DOI exists): {title[:50]}...")
                return result

        # Check for existing document with same pdf_hash (skip if same file)
        cur.execute("SELECT doc_id, title FROM documents WHERE pdf_hash = %s", (pdf_hash,))
        existing = cur.fetchone()
        if existing:
            result['status'] = 'skipped'
            result['error'] = f'Same PDF already ingested: {existing[1][:50]}...'
            logger.info(f"Skipped (same file): {title[:50]}...")
            return result

        # Chunk text
        chunks = chunk_text(parse_result.text)
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

        # Store in database (cursor already created for dedup checks)
        # Parse authors string into array (Zotero format: "LastName, First; LastName2, First2")
        authors_array = [a.strip() for a in authors.split(';') if a.strip()] if authors else None

        # Parse year as integer
        year_int = int(year) if year and year.isdigit() else None

        # Upsert document (with full metadata)
        cur.execute("""
            INSERT INTO documents (
                doc_id, title, title_hash, doi, pdf_hash, pdf_path, ingest_batch,
                authors, year, abstract, venue, zotero_key, source_method
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                doi = COALESCE(EXCLUDED.doi, documents.doi),
                pdf_hash = COALESCE(EXCLUDED.pdf_hash, documents.pdf_hash),
                pdf_path = EXCLUDED.pdf_path,
                authors = COALESCE(EXCLUDED.authors, documents.authors),
                year = COALESCE(EXCLUDED.year, documents.year),
                abstract = COALESCE(EXCLUDED.abstract, documents.abstract),
                venue = COALESCE(EXCLUDED.venue, documents.venue),
                zotero_key = COALESCE(EXCLUDED.zotero_key, documents.zotero_key),
                source_method = EXCLUDED.source_method,
                updated_at = NOW()
            RETURNING doc_id
        """, (
            str(doc_id), title, get_title_hash(title), doi, pdf_hash, str(pdf_path), batch_name,
            authors_array, year_int, abstract or None, venue or None, zotero_key or None, metadata_source
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
        if detect_assets:
            detector = AssetDetector(conn)
            passages = [{'passage_id': str(uuid.uuid5(doc_id, f"passage:{i}")),
                        'passage_text': c['content']} for i, c in enumerate(chunks)]
            asset_counts = detector.detect_and_store(str(doc_id), passages)
            result['assets'] = asset_counts

        result['status'] = 'success'
        logger.info(f"Ingested: {title[:50]}... ({len(chunks)} passages)")

    except Exception as e:
        logger.error(f"Error ingesting {pdf_path}: {e}")
        result['status'] = 'error'
        result['error'] = str(e)
        conn.rollback()

    finally:
        if close_conn:
            conn.close()

    return result


def ingest_batch(
    pdf_paths: List[Path],
    workers: int = 4,
    batch_name: str = None
) -> dict:
    """Ingest multiple PDFs in parallel."""
    import time
    from datetime import datetime

    if batch_name is None:
        batch_name = f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    start = time.time()
    results = []
    succeeded = 0
    failed = 0

    logger.info(f"Starting batch '{batch_name}' with {len(pdf_paths)} files, {workers} workers")

    # Pre-load embedder (one-time GPU init)
    if workers > 0:
        get_embedder()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(ingest_single_pdf, p, None, True, True, batch_name): p
            for p in pdf_paths
        }

        for future in as_completed(futures):
            pdf_path = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result['status'] == 'success':
                    succeeded += 1
                else:
                    failed += 1
                    logger.warning(f"Failed: {pdf_path.name} - {result.get('error')}")
            except Exception as e:
                failed += 1
                logger.error(f"Exception: {pdf_path.name} - {e}")

    elapsed = time.time() - start

    summary = {
        'batch_name': batch_name,
        'total': len(pdf_paths),
        'succeeded': succeeded,
        'failed': failed,
        'elapsed_seconds': round(elapsed, 1),
        'results': results
    }

    logger.info(f"Batch complete: {succeeded}/{len(pdf_paths)} succeeded in {elapsed:.1f}s")
    return summary


def main():
    global _zotero_metadata

    parser = argparse.ArgumentParser(description='Ingest PDFs into Polymath')
    parser.add_argument('pdfs', nargs='+', help='PDF files or directories to ingest')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Parallel workers')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embedding computation')
    parser.add_argument('--no-assets', action='store_true', help='Skip asset detection')
    parser.add_argument('--batch-name', help='Name for this batch')
    parser.add_argument('--zotero-csv', help='Zotero CSV with metadata (from prepare_zotero_ingest.py)')
    parser.add_argument('--recursive', '-r', action='store_true', help='Recursively search directories for PDFs')
    args = parser.parse_args()

    # Load Zotero metadata if provided
    if args.zotero_csv:
        _zotero_metadata = load_zotero_metadata(args.zotero_csv)

    # Expand paths
    pdf_paths = []
    for p in args.pdfs:
        path = Path(p)
        if path.is_dir():
            if args.recursive:
                pdf_paths.extend(path.rglob('*.pdf'))
            else:
                pdf_paths.extend(path.glob('*.pdf'))
        elif path.exists():
            pdf_paths.append(path)
        else:
            logger.warning(f"Not found: {p}")

    if not pdf_paths:
        logger.error("No PDF files found")
        return

    if len(pdf_paths) == 1:
        result = ingest_single_pdf(
            pdf_paths[0],
            compute_embeddings=not args.no_embeddings,
            detect_assets=not args.no_assets,
            batch_name=args.batch_name
        )
        print(f"\nResult: {result['status']}")
        if result['status'] == 'success':
            print(f"  Doc ID: {result['doc_id']}")
            print(f"  Title: {result['title'][:60]}...")
            print(f"  Passages: {result['passages']}")
    else:
        summary = ingest_batch(pdf_paths, args.workers, args.batch_name)
        print(f"\n{'='*60}")
        print(f"BATCH SUMMARY")
        print(f"{'='*60}")
        print(f"Total: {summary['total']}")
        print(f"Succeeded: {summary['succeeded']}")
        print(f"Failed: {summary['failed']}")
        print(f"Time: {summary['elapsed_seconds']}s")


if __name__ == '__main__':
    main()
