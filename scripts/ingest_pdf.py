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

# Lazy load embedder (GPU resource)
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from lib.embeddings.bge_m3 import BGEM3Embedder
        _embedder = BGEM3Embedder()
    return _embedder


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

        # Extract title from first line or filename
        lines = parse_result.text.strip().split('\n')
        title = lines[0][:200] if lines else pdf_path.stem

        # Generate doc_id
        doc_id = get_doc_id(title)
        result['doc_id'] = str(doc_id)
        result['title'] = title

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

        # Store in database
        cur = conn.cursor()

        # Upsert document
        cur.execute("""
            INSERT INTO documents (doc_id, title, title_hash, pdf_path, ingest_batch)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO UPDATE SET
                title = EXCLUDED.title,
                pdf_path = EXCLUDED.pdf_path,
                updated_at = NOW()
            RETURNING doc_id
        """, (str(doc_id), title, get_title_hash(title), str(pdf_path), batch_name))

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
    parser = argparse.ArgumentParser(description='Ingest PDFs into Polymath')
    parser.add_argument('pdfs', nargs='+', help='PDF files to ingest')
    parser.add_argument('--workers', '-w', type=int, default=4, help='Parallel workers')
    parser.add_argument('--no-embeddings', action='store_true', help='Skip embedding computation')
    parser.add_argument('--no-assets', action='store_true', help='Skip asset detection')
    parser.add_argument('--batch-name', help='Name for this batch')
    args = parser.parse_args()

    # Expand paths
    pdf_paths = []
    for p in args.pdfs:
        path = Path(p)
        if path.is_dir():
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
