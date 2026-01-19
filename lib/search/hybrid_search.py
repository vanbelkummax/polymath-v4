"""
Hybrid Search for Polymath v4

Combines vector similarity with BM25 for optimal retrieval.
Includes GraphRAG query expansion via Neo4j concept graph.
"""

import logging
import re
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

import numpy as np

from lib.config import config
from lib.embeddings.bge_m3 import BGEEmbedder
from lib.db.postgres import get_db_connection

logger = logging.getLogger(__name__)

# Neo4j connection (lazy loaded)
_neo4j_driver = None


def _get_neo4j_driver():
    """Get or create Neo4j driver."""
    global _neo4j_driver
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase
            _neo4j_driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
    return _neo4j_driver


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

    def _expand_query_with_graph(
        self,
        query: str,
        max_expansions: int = None,
        min_co_occurrences: int = None,
        conn=None
    ) -> Tuple[str, List[str]]:
        """
        Expand query using concept co-occurrence from passage_concepts table.

        Finds related concepts via:
        - Direct match to known concepts in passage_concepts
        - Co-occurrence: concepts appearing in same passages

        Args:
            query: Original search query
            max_expansions: Max number of expansion terms to add (default: config)
            min_co_occurrences: Minimum co-occurrence count for inclusion (default: config)
            conn: Optional database connection

        Returns:
            Tuple of (expanded_query, expansion_terms)
        """
        # Use config defaults if not specified
        if max_expansions is None:
            max_expansions = config.SEARCH_GRAPHRAG_MAX_EXPANSIONS
        if min_co_occurrences is None:
            min_co_occurrences = config.SEARCH_GRAPHRAG_MIN_COOCCURRENCE

        should_close = conn is None
        if conn is None:
            try:
                conn = self._get_connection()
            except Exception as e:
                logger.warning(f"GraphRAG expansion failed (db): {e}")
                return query, []

        try:
            # Extract query terms (normalize)
            terms = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b', query.lower())
            terms = set(terms)

            if not terms:
                return query, []

            # Build LIKE patterns for concept matching
            patterns = []
            for term in terms:
                patterns.append(f"%{term}%")
                patterns.append(f"%{term.replace('-', '_')}%")

            cur = conn.cursor()

            # Find concepts matching query terms and their co-occurring concepts
            # This uses passage_concepts to find concepts that appear together
            cur.execute("""
                WITH query_concepts AS (
                    -- Find concepts matching query terms
                    SELECT DISTINCT concept_name, passage_id
                    FROM passage_concepts
                    WHERE concept_name ILIKE ANY(%s)
                    LIMIT 1000
                ),
                co_occurring AS (
                    -- Find concepts co-occurring in same passages
                    SELECT
                        pc.concept_name,
                        pc.concept_type,
                        COUNT(DISTINCT pc.passage_id) as co_occurrences
                    FROM passage_concepts pc
                    JOIN query_concepts qc ON pc.passage_id = qc.passage_id
                    WHERE pc.concept_name NOT ILIKE ANY(%s)
                    AND pc.confidence > 0.5
                    GROUP BY pc.concept_name, pc.concept_type
                    HAVING COUNT(DISTINCT pc.passage_id) >= %s
                    ORDER BY co_occurrences DESC
                    LIMIT %s
                )
                SELECT concept_name, concept_type, co_occurrences
                FROM co_occurring
                ORDER BY co_occurrences DESC
            """, (patterns, patterns, min_co_occurrences, max_expansions * 2))

            rows = cur.fetchall()

            # Collect expansion terms with scores
            expansion_terms = []
            for concept_name, concept_type, count in rows:
                # Skip very generic concepts
                if concept_name.lower() in {'method', 'model', 'data', 'analysis', 'result', 'study'}:
                    continue
                expansion_terms.append(concept_name)
                if len(expansion_terms) >= max_expansions:
                    break

            if expansion_terms:
                # Build expanded query with OR
                # Replace underscores with spaces for BM25
                expanded_terms = [t.replace('_', ' ') for t in expansion_terms]
                expanded = f"{query} OR " + " OR ".join(f'"{t}"' for t in expanded_terms)
                logger.info(f"GraphRAG expanded: '{query}' → +{len(expansion_terms)} terms: {expansion_terms[:3]}...")
                return expanded, expansion_terms

        except Exception as e:
            logger.warning(f"GraphRAG expansion failed: {e}")

        finally:
            if should_close and conn:
                conn.close()

        return query, []

    def _get_connection(self):
        """Get a database connection (standalone, caller must close)."""
        return get_db_connection()

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
        vector_weight: float = None,
        rerank: bool = None,
        graph_expand: bool = False,
        conn=None
    ) -> List[SearchResult]:
        """
        Hybrid search combining vector and BM25.

        Args:
            query: Search query
            n: Number of results
            vector_weight: Weight for vector scores (0-1), default from config
            rerank: Whether to rerank results (default: self.rerank_enabled)
            graph_expand: Whether to use GraphRAG query expansion via Neo4j
        """
        # Use config defaults
        if vector_weight is None:
            vector_weight = config.SEARCH_VECTOR_WEIGHT
        if rerank is None:
            rerank = self.rerank_enabled

        # GraphRAG expansion
        original_query = query
        expansion_terms = []
        if graph_expand:
            query, expansion_terms = self._expand_query_with_graph(query)

        should_close = conn is None
        if conn is None:
            conn = self._get_connection()

        # Get more candidates for fusion (configurable multiplier)
        k = n * config.SEARCH_CANDIDATE_MULTIPLIER

        # Use original query for vector search (embeddings capture semantics)
        # Use expanded query for BM25 (lexical expansion helps)
        vector_results = self.vector_search(original_query, k, conn)
        bm25_results = self.bm25_search(query, k, conn)

        # Reciprocal Rank Fusion with configurable k parameter
        rrf_k = config.SEARCH_RRF_K
        scores = {}
        passages = {}

        for i, r in enumerate(vector_results):
            rrf_score = vector_weight / (rrf_k + i)
            scores[r.passage_id] = scores.get(r.passage_id, 0) + rrf_score
            passages[r.passage_id] = r

        bm25_weight = 1 - vector_weight
        for i, r in enumerate(bm25_results):
            rrf_score = bm25_weight / (rrf_k + i)
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


# Global searcher instance for fast repeated queries
_global_searcher: Optional[HybridSearcher] = None


def warmup(rerank: bool = True) -> HybridSearcher:
    """
    Pre-warm the search models to avoid first-query latency.

    Call this at application startup or before batch searches.
    Returns a ready-to-use HybridSearcher.

    Example:
        # At startup
        searcher = warmup()

        # Fast queries (no model loading)
        results = searcher.hybrid_search("spatial transcriptomics")
    """
    global _global_searcher

    logger.info("Warming up search models...")

    if _global_searcher is None:
        _global_searcher = HybridSearcher(rerank=rerank)

    # Pre-load embedder by encoding a dummy query
    embedder = _global_searcher._get_embedder()
    _ = embedder.model  # Force model loading
    _ = embedder.embed_single("warmup query")  # Force CUDA initialization

    # Pre-load reranker if enabled
    if rerank:
        _global_searcher._get_reranker()

    logger.info("Search models warmed up and ready")
    return _global_searcher


def get_searcher(rerank: bool = True) -> HybridSearcher:
    """Get the global searcher instance, warming up if needed."""
    global _global_searcher
    if _global_searcher is None:
        return warmup(rerank=rerank)
    return _global_searcher
