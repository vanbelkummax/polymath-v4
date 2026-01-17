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
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
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

        self._conn = None
        self._embedder = None
        self._asset_detector = None
        self._neo4j = None

    def _get_connection(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(config.POSTGRES_DSN)
        return self._conn

    def _get_embedder(self) -> BGEEmbedder:
        if self._embedder is None:
            self._embedder = BGEEmbedder()
        return self._embedder

    def _get_asset_detector(self) -> AssetDetector:
        if self._asset_detector is None:
            self._asset_detector = AssetDetector()
        return self._asset_detector

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
                INSERT INTO documents (doc_id, title, authors, title_hash, file_path, content_hash, ingested_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (doc_id) DO NOTHING
                RETURNING doc_id
            """, (
                doc_id,
                title[:500],
                authors[:10] if authors else None,
                title_hash,
                str(pdf_path),
                content_hash
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
                        chunk_index, header, char_start, char_end, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (passage_id) DO NOTHING
                """, (
                    passage_id,
                    doc_id,
                    chunk['content'],
                    chunk.get('page_num'),
                    i,
                    chunk.get('header'),
                    chunk.get('char_start'),
                    chunk.get('char_end'),
                    embedding
                ))
                passages_added += 1

            conn.commit()

            # 8. Detect assets (GitHub repos, HF models)
            assets_detected = 0
            if self.detect_assets:
                try:
                    detector = self._get_asset_detector()
                    for i, chunk in enumerate(chunks):
                        assets = detector.detect_all(chunk['content'])

                        for asset in assets.get('github', []):
                            cur.execute("""
                                INSERT INTO repo_queue (repo_url, source_doc_id, source_passage_id, priority)
                                VALUES (%s, %s, %s, %s)
                                ON CONFLICT (repo_url) DO UPDATE SET
                                    priority = GREATEST(repo_queue.priority, EXCLUDED.priority)
                            """, (
                                asset['url'],
                                doc_id,
                                passage_ids[i] if i < len(passage_ids) else None,
                                asset.get('priority', 50)
                            ))
                            assets_detected += 1

                        for asset in assets.get('huggingface', []):
                            cur.execute("""
                                INSERT INTO hf_model_mentions (model_id, doc_id, passage_id)
                                VALUES (%s, %s, %s)
                                ON CONFLICT DO NOTHING
                            """, (
                                asset['model_id'],
                                doc_id,
                                passage_ids[i] if i < len(passage_ids) else None
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
        cur.execute("""
            INSERT INTO ingest_runs (run_id, run_type, metadata)
            VALUES (%s, 'pdf', %s)
        """, (run_id, {'batch_name': batch_name, 'pattern': pattern}))
        conn.commit()

        logger.info(f"Starting batch ingestion: {len(pdf_files)} files")

        # Process files
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.ingest_pdf, str(f), batch_name): f
                for f in pdf_files
            }

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    pdf_path = futures[future]
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
        cur.execute("""
            UPDATE ingest_runs SET
                completed_at = NOW(),
                status = 'completed',
                items_processed = %s,
                items_failed = %s
            WHERE run_id = %s
        """, (successful, failed, run_id))
        conn.commit()

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
        if self._conn and not self._conn.closed:
            self._conn.close()
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
