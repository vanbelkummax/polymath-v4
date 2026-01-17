"""
Hybrid Search for Polymath v4

Combines vector similarity with BM25 for optimal retrieval.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import psycopg2

from lib.config import config
from lib.embeddings.bge_m3 import BGEEmbedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata."""
    passage_id: str
    passage_text: str
    doc_id: str
    title: str
    score: float
    source: str  # 'vector', 'bm25', or 'hybrid'


class HybridSearcher:
    """Hybrid search combining vector similarity and BM25."""

    def __init__(self, rerank: bool = True):
        self.embedder = None
        self.reranker = None
        self.rerank_enabled = rerank

    def _get_embedder(self) -> BGEEmbedder:
        if self.embedder is None:
            self.embedder = BGEEmbedder()
        return self.embedder

    def _get_reranker(self):
        if self.reranker is None and self.rerank_enabled:
            try:
                from sentence_transformers import CrossEncoder
                self.reranker = CrossEncoder(config.RERANKER_MODEL, device='cuda')
            except Exception as e:
                logger.warning(f"Could not load reranker: {e}")
                self.rerank_enabled = False
        return self.reranker

    def _get_connection(self):
        return psycopg2.connect(config.POSTGRES_DSN)

    def vector_search(
        self,
        query: str,
        n: int = 20,
        conn=None
    ) -> List[SearchResult]:
        """Pure vector similarity search."""
        embedder = self._get_embedder()
        query_embedding = embedder.embed_single(query)

        should_close = conn is None
        if conn is None:
            conn = self._get_connection()

        cur = conn.cursor()

        # pgvector cosine similarity search
        cur.execute("""
            SELECT
                p.passage_id::text,
                p.passage_text,
                d.doc_id::text,
                d.title,
                1 - (p.embedding <=> %s::vector) as similarity
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE p.embedding IS NOT NULL
            AND p.is_superseded = FALSE
            ORDER BY p.embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding.tolist(), query_embedding.tolist(), n))

        results = [
            SearchResult(
                passage_id=row[0],
                passage_text=row[1],
                doc_id=row[2],
                title=row[3],
                score=float(row[4]),
                source='vector'
            )
            for row in cur.fetchall()
        ]

        if should_close:
            conn.close()

        return results

    def bm25_search(
        self,
        query: str,
        n: int = 20,
        conn=None
    ) -> List[SearchResult]:
        """BM25 full-text search using PostgreSQL ts_rank."""
        should_close = conn is None
        if conn is None:
            conn = self._get_connection()

        cur = conn.cursor()

        # PostgreSQL full-text search
        cur.execute("""
            SELECT
                p.passage_id::text,
                p.passage_text,
                d.doc_id::text,
                d.title,
                ts_rank(to_tsvector('english', p.passage_text), plainto_tsquery('english', %s)) as rank
            FROM passages p
            JOIN documents d ON p.doc_id = d.doc_id
            WHERE to_tsvector('english', p.passage_text) @@ plainto_tsquery('english', %s)
            AND p.is_superseded = FALSE
            ORDER BY rank DESC
            LIMIT %s
        """, (query, query, n))

        results = [
            SearchResult(
                passage_id=row[0],
                passage_text=row[1],
                doc_id=row[2],
                title=row[3],
                score=float(row[4]),
                source='bm25'
            )
            for row in cur.fetchall()
        ]

        if should_close:
            conn.close()

        return results

    def hybrid_search(
        self,
        query: str,
        n: int = 20,
        vector_weight: float = 0.7,
        rerank: bool = None,
        conn=None
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query
            n: Number of results
            vector_weight: Weight for vector scores (0-1)
            rerank: Whether to rerank results (default: self.rerank_enabled)
        """
        if rerank is None:
            rerank = self.rerank_enabled

        should_close = conn is None
        if conn is None:
            conn = self._get_connection()

        # Get more candidates for fusion
        k = n * 3

        vector_results = self.vector_search(query, k, conn)
        bm25_results = self.bm25_search(query, k, conn)

        # Reciprocal Rank Fusion
        scores = {}
        passages = {}

        for i, r in enumerate(vector_results):
            rrf_score = vector_weight / (60 + i)
            scores[r.passage_id] = scores.get(r.passage_id, 0) + rrf_score
            passages[r.passage_id] = r

        bm25_weight = 1 - vector_weight
        for i, r in enumerate(bm25_results):
            rrf_score = bm25_weight / (60 + i)
            scores[r.passage_id] = scores.get(r.passage_id, 0) + rrf_score
            if r.passage_id not in passages:
                passages[r.passage_id] = r

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:n * 2]

        results = [
            SearchResult(
                passage_id=pid,
                passage_text=passages[pid].passage_text,
                doc_id=passages[pid].doc_id,
                title=passages[pid].title,
                score=scores[pid],
                source='hybrid'
            )
            for pid in sorted_ids
        ]

        # Rerank if enabled
        if rerank and results:
            results = self._rerank(query, results, n)
        else:
            results = results[:n]

        if should_close:
            conn.close()

        return results

    def _rerank(
        self,
        query: str,
        results: List[SearchResult],
        n: int
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder."""
        reranker = self._get_reranker()
        if reranker is None:
            return results[:n]

        pairs = [(query, r.passage_text[:1000]) for r in results]
        scores = reranker.predict(pairs)

        # Sort by reranker scores
        ranked = sorted(
            zip(results, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            SearchResult(
                passage_id=r.passage_id,
                passage_text=r.passage_text,
                doc_id=r.doc_id,
                title=r.title,
                score=float(s),
                source='reranked'
            )
            for r, s in ranked[:n]
        ]

    def concept_search(
        self,
        concept: str,
        concept_type: Optional[str] = None,
        n: int = 20,
        conn=None
    ) -> List[SearchResult]:
        """Search by extracted concept."""
        should_close = conn is None
        if conn is None:
            conn = self._get_connection()

        cur = conn.cursor()

        if concept_type:
            cur.execute("""
                SELECT
                    p.passage_id::text,
                    p.passage_text,
                    d.doc_id::text,
                    d.title,
                    pc.confidence
                FROM passage_concepts pc
                JOIN passages p ON pc.passage_id = p.passage_id
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE pc.concept_name ILIKE %s
                AND pc.concept_type = %s
                AND p.is_superseded = FALSE
                ORDER BY pc.confidence DESC
                LIMIT %s
            """, (f'%{concept}%', concept_type, n))
        else:
            cur.execute("""
                SELECT
                    p.passage_id::text,
                    p.passage_text,
                    d.doc_id::text,
                    d.title,
                    pc.confidence
                FROM passage_concepts pc
                JOIN passages p ON pc.passage_id = p.passage_id
                JOIN documents d ON p.doc_id = d.doc_id
                WHERE pc.concept_name ILIKE %s
                AND p.is_superseded = FALSE
                ORDER BY pc.confidence DESC
                LIMIT %s
            """, (f'%{concept}%', n))

        results = [
            SearchResult(
                passage_id=row[0],
                passage_text=row[1],
                doc_id=row[2],
                title=row[3],
                score=float(row[4]),
                source='concept'
            )
            for row in cur.fetchall()
        ]

        if should_close:
            conn.close()

        return results


def search(query: str, n: int = 10, rerank: bool = True) -> List[Dict]:
    """Convenience function for quick searches."""
    searcher = HybridSearcher(rerank=rerank)
    results = searcher.hybrid_search(query, n)
    return [
        {
            'passage_id': r.passage_id,
            'text': r.passage_text,
            'title': r.title,
            'score': r.score
        }
        for r in results
    ]
