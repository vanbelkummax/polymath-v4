#!/usr/bin/env python3
"""
Unified Ingestion Pipeline for Polymath v4

PostgreSQL-first with local BGE-M3 embeddings.
Optional Neo4j sync for knowledge graph.

Usage:
    ingestor = UnifiedIngestor()
    result = ingestor.ingest_pdf("/path/to/paper.pdf")
    batch_result = ingestor.ingest_directory("/path/to/pdfs/")
"""

import hashlib
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import psycopg2

from lib.config import config
from lib.ingest.pdf_parser import PDFParser
from lib.ingest.chunking import chunk_text
from lib.ingest.asset_detector import AssetDetector
from lib.embeddings.bge_m3 import BGEEmbedder

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of ingesting a single document."""
    doc_id: str
    title: str
    passages_added: int
    concepts_extracted: int = 0
    assets_detected: int = 0
    neo4j_synced: bool = False
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchResult:
    """Result of batch ingestion."""
    total_files: int
    successful: int
    failed: int
    passages_added: int
    run_id: Optional[str] = None
    results: List[IngestResult] = field(default_factory=list)


class UnifiedIngestor:
    """
    PostgreSQL-first ingestion with local embeddings.

    Features:
    - PDF parsing with PyMuPDF
    - Smart chunking with header detection
    - BGE-M3 embeddings (local GPU)
    - Asset detection (GitHub, HuggingFace, DOI)
    - Optional Neo4j sync
    """

    def __init__(
        self,
        compute_embeddings: bool = True,
        detect_assets: bool = True,
        sync_neo4j: bool = False
    ):
        self.compute_embeddings = compute_embeddings
        self.detect_assets = detect_assets
        self.sync_neo4j = sync_neo4j

        # Shared resources (thread-safe or re-created per task)
        self._embedder = None
        self._neo4j = None

    def _get_connection(self):
        """Create a new connection (each task gets its own for thread safety)."""
        return psycopg2.connect(config.POSTGRES_DSN)

    def _get_embedder(self) -> BGEEmbedder:
        """Get shared embedder (thread-safe for inference)."""
        if self._embedder is None:
            self._embedder = BGEEmbedder()
        return self._embedder

    def _get_neo4j(self):
        if self._neo4j is None and self.sync_neo4j:
            try:
                from neo4j import GraphDatabase
                self._neo4j = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
                )
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j: {e}")
        return self._neo4j

    def _compute_doc_id(self, title: str, content_hash: str) -> str:
        """Generate deterministic doc_id from title + content."""
        combined = f"{title.lower().strip()}:{content_hash}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, combined))

    def _compute_title_hash(self, title: str) -> str:
        """Compute normalized title hash for deduplication."""
        normalized = ''.join(c.lower() for c in title if c.isalnum())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def ingest_pdf(self, pdf_path: str, batch_name: Optional[str] = None) -> IngestResult:
        """
        Ingest a single PDF into PostgreSQL.

        Flow:
        1. Parse PDF → Extract text and metadata
        2. Chunk → Split into passages
        3. Embed → Compute BGE-M3 vectors
        4. Store → Save to PostgreSQL
        5. Detect → Find GitHub/HF assets
        6. Sync → Optional Neo4j update
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return IngestResult(
                doc_id='',
                title=str(pdf_path),
                passages_added=0,
                errors=[f"File not found: {pdf_path}"]
            )

        # Create connection for this task (thread-safe)
        conn = self._get_connection()
        cur = conn.cursor()
        errors = []

        try:
            # 1. Parse PDF
            parser = PDFParser()
            doc = parser.parse(str(pdf_path))

            title = doc.get('title', pdf_path.stem)
            authors = doc.get('authors', [])
            text = doc.get('text', '')

            if not text or len(text) < 100:
                return IngestResult(
                    doc_id='',
                    title=title,
                    passages_added=0,
                    errors=["PDF has no extractable text"]
                )

            # 2. Compute identifiers
            content_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
            doc_id = self._compute_doc_id(title, content_hash)
            title_hash = self._compute_title_hash(title)

            # 3. Check for duplicates
            cur.execute("""
                SELECT doc_id FROM documents WHERE title_hash = %s
            """, (title_hash,))
            existing = cur.fetchone()

            if existing:
                logger.info(f"Document already exists: {title[:50]}")
                return IngestResult(
                    doc_id=str(existing[0]),
                    title=title,
                    passages_added=0,
                    errors=["Document already ingested (duplicate)"]
                )

            # 4. Insert document
            cur.execute("""
                INSERT INTO documents (doc_id, title, authors, title_hash, pdf_path, ingest_batch)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO NOTHING
                RETURNING doc_id
            """, (
                doc_id,
                title[:500],
                authors[:10] if authors else None,
                title_hash,
                str(pdf_path),
                batch_name
            ))

            # 5. Chunk text
            chunks = chunk_text(text, max_size=1500, overlap=200)

            if not chunks:
                conn.commit()
                return IngestResult(
                    doc_id=doc_id,
                    title=title,
                    passages_added=0,
                    errors=["No chunks extracted"]
                )

            # 6. Compute embeddings
            embeddings = None
            if self.compute_embeddings:
                try:
                    embedder = self._get_embedder()
                    texts = [c['content'] for c in chunks]
                    embeddings = embedder.embed_batch(texts)
                except Exception as e:
                    errors.append(f"Embedding error: {e}")
                    logger.warning(f"Could not compute embeddings: {e}")

            # 7. Insert passages
            passages_added = 0
            passage_ids = []

            for i, chunk in enumerate(chunks):
                passage_id = str(uuid.uuid4())
                passage_ids.append(passage_id)

                embedding = embeddings[i].tolist() if embeddings is not None else None

                cur.execute("""
                    INSERT INTO passages (
                        passage_id, doc_id, passage_text, page_num,
                        passage_index, section, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (passage_id) DO NOTHING
                """, (
                    passage_id,
                    doc_id,
                    chunk['content'],
                    chunk.get('page_num'),
                    i,
                    chunk.get('header'),
                    embedding
                ))
                passages_added += 1

            conn.commit()

            # 8. Detect assets (GitHub repos, HF models)
            assets_detected = 0
            if self.detect_assets:
                try:
                    detector = AssetDetector()
                    for i, chunk in enumerate(chunks):
                        # Use detect_from_text for string input
                        passage_id = passage_ids[i] if i < len(passage_ids) else None
                        assets = detector.detect_from_text(chunk['content'], passage_id or '')

                        for asset in assets.get('github', []):
                            # Extract owner/name from identifier
                            identifier = asset.identifier
                            owner = asset.extra.get('owner', '')
                            name = asset.extra.get('name', '')

                            cur.execute("""
                                INSERT INTO repo_queue (repo_url, owner, repo_name, first_seen_doc_id, source, priority)
                                VALUES (%s, %s, %s, %s, 'paper_detection', 5)
                                ON CONFLICT (repo_url) DO UPDATE SET
                                    priority = GREATEST(repo_queue.priority + 1, EXCLUDED.priority),
                                    source_doc_count = COALESCE(repo_queue.source_doc_count, 0) + 1
                            """, (
                                identifier,
                                owner,
                                name,
                                doc_id
                            ))
                            assets_detected += 1

                        for asset in assets.get('huggingface', []):
                            model_id_raw = asset.identifier
                            cur.execute("""
                                INSERT INTO hf_model_mentions (model_id_raw, doc_id, passage_id, context)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (
                                model_id_raw,
                                doc_id,
                                passage_id,
                                asset.context[:500] if asset.context else None
                            ))
                            assets_detected += 1

                    conn.commit()
                except Exception as e:
                    errors.append(f"Asset detection error: {e}")
                    logger.warning(f"Asset detection failed: {e}")

            # 9. Sync to Neo4j (optional)
            neo4j_synced = False
            if self.sync_neo4j:
                try:
                    driver = self._get_neo4j()
                    if driver:
                        with driver.session() as session:
                            session.run("""
                                MERGE (d:Document {doc_id: $doc_id})
                                SET d.title = $title,
                                    d.authors = $authors,
                                    d.passages = $passages,
                                    d.updated_at = datetime()
                            """, {
                                'doc_id': doc_id,
                                'title': title,
                                'authors': authors,
                                'passages': passages_added
                            })
                        neo4j_synced = True
                except Exception as e:
                    errors.append(f"Neo4j sync error: {e}")
                    logger.warning(f"Neo4j sync failed: {e}")

            return IngestResult(
                doc_id=doc_id,
                title=title,
                passages_added=passages_added,
                assets_detected=assets_detected,
                neo4j_synced=neo4j_synced,
                errors=errors
            )

        except Exception as e:
            conn.rollback()
            logger.error(f"Ingestion failed for {pdf_path}: {e}")
            return IngestResult(
                doc_id='',
                title=str(pdf_path),
                passages_added=0,
                errors=[str(e)]
            )
        finally:
            # Always close connection for this task
            conn.close()

    def ingest_directory(
        self,
        directory: str,
        pattern: str = "*.pdf",
        workers: int = 4,
        batch_name: Optional[str] = None
    ) -> BatchResult:
        """
        Ingest all PDFs in a directory.

        Args:
            directory: Path to directory
            pattern: Glob pattern for files
            workers: Number of parallel workers
            batch_name: Optional batch identifier
        """
        directory = Path(directory)
        pdf_files = list(directory.glob(pattern))

        if not pdf_files:
            return BatchResult(
                total_files=0,
                successful=0,
                failed=0,
                passages_added=0
            )

        run_id = str(uuid.uuid4())
        results = []

        # Track in database
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO ingest_runs (run_id, run_type, metadata)
                VALUES (%s, 'pdf', %s)
            """, (run_id, json.dumps({'batch_name': batch_name, 'pattern': pattern})))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Starting batch ingestion: {len(pdf_files)} files")

        # Pre-load embedder (single instance, shared)
        if self.compute_embeddings:
            self._get_embedder()

        # Process files with thread pool (each task gets its own connection)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.ingest_pdf, str(f), batch_name): f
                for f in pdf_files
            }

            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(IngestResult(
                        doc_id='',
                        title=str(pdf_path),
                        passages_added=0,
                        errors=[str(e)]
                    ))

        # Summarize
        successful = sum(1 for r in results if r.passages_added > 0)
        failed = len(results) - successful
        passages_added = sum(r.passages_added for r in results)

        # Update run record
        conn = self._get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                UPDATE ingest_runs SET
                    completed_at = NOW(),
                    status = 'completed',
                    items_processed = %s,
                    items_failed = %s
                WHERE run_id = %s
            """, (successful, failed, run_id))
            conn.commit()
        finally:
            conn.close()

        return BatchResult(
            total_files=len(pdf_files),
            successful=successful,
            failed=failed,
            passages_added=passages_added,
            run_id=run_id,
            results=results
        )

    def close(self):
        """Clean up resources."""
        if self._neo4j:
            self._neo4j.close()


def ingest_pdf(pdf_path: str, **kwargs) -> IngestResult:
    """Convenience function for single PDF ingestion."""
    ingestor = UnifiedIngestor(**kwargs)
    try:
        return ingestor.ingest_pdf(pdf_path)
    finally:
        ingestor.close()


def ingest_directory(directory: str, **kwargs) -> BatchResult:
    """Convenience function for directory ingestion."""
    ingestor = UnifiedIngestor(**kwargs)
    try:
        return ingestor.ingest_directory(directory, **kwargs)
    finally:
        ingestor.close()
