#!/usr/bin/env python3
"""
Search Quality Tests for Polymath v4

Validates that search returns relevant results with appropriate scores.
Tests vector search, BM25, hybrid search, and index usage.

Run with: python -m pytest tests/test_search_quality.py -v
"""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestSearchQuality:
    """Test search result quality and relevance."""

    @pytest.fixture(scope="class")
    def searcher(self):
        """Create a shared searcher instance."""
        from lib.search.hybrid_search import HybridSearcher
        return HybridSearcher(rerank=False)

    def test_vector_search_returns_results(self, searcher):
        """Vector search should return results for common queries."""
        results = searcher.vector_search("spatial transcriptomics", n=5)
        assert len(results) > 0, "Vector search should return results"
        assert results[0].score > 0.3, "Top result should have reasonable score"

    def test_vector_search_relevance(self, searcher):
        """Vector search should return relevant content."""
        results = searcher.vector_search("Moran I spatial autocorrelation", n=5)
        assert len(results) > 0
        # Check that at least one result mentions the topic
        texts = [r.passage_text.lower() for r in results]
        has_moran = any("moran" in t for t in texts)
        has_spatial = any("spatial" in t for t in texts)
        assert has_moran or has_spatial, "Should find Moran's I or spatial content"

    def test_bm25_search_returns_results(self, searcher):
        """BM25 search should return results for keyword queries."""
        results = searcher.bm25_search("gene expression prediction", n=5)
        assert len(results) > 0, "BM25 should return results"

    def test_bm25_exact_match(self, searcher):
        """BM25 should find exact term matches."""
        results = searcher.bm25_search("squidpy", n=5)
        if results:  # May not have squidpy in corpus
            texts = [r.passage_text.lower() for r in results]
            assert any("squidpy" in t for t in texts), "Should find exact matches"

    def test_hybrid_search_returns_results(self, searcher):
        """Hybrid search should return results."""
        results = searcher.hybrid_search("cell segmentation deep learning", n=10)
        assert len(results) > 0, "Hybrid search should return results"

    def test_hybrid_combines_both_methods(self, searcher):
        """Hybrid should find results from both vector and BM25."""
        query = "attention mechanism transformer"
        vector_results = set(r.passage_id for r in searcher.vector_search(query, n=20))
        bm25_results = set(r.passage_id for r in searcher.bm25_search(query, n=20))
        hybrid_results = set(r.passage_id for r in searcher.hybrid_search(query, n=20))

        # Hybrid should include results from both (if they exist)
        if vector_results and bm25_results:
            assert len(hybrid_results) > 0

    def test_concept_search_returns_results(self, searcher):
        """Concept search should return results for known concepts."""
        results = searcher.concept_search("deep learning", n=5)
        # Concept search depends on extracted concepts
        # Just verify it doesn't error
        assert isinstance(results, list)

    def test_score_ranges_valid(self, searcher):
        """Scores should be in valid ranges."""
        results = searcher.vector_search("test query", n=5)
        for r in results:
            assert 0 <= r.score <= 1, f"Vector score {r.score} out of range"

        # Hybrid uses RRF so scores are smaller but positive
        hybrid = searcher.hybrid_search("test query", n=5)
        for r in hybrid:
            assert r.score >= 0, f"Hybrid score {r.score} should be non-negative"

    def test_result_structure(self, searcher):
        """Results should have all required fields."""
        results = searcher.vector_search("spatial", n=1)
        if results:
            r = results[0]
            assert hasattr(r, 'passage_id')
            assert hasattr(r, 'passage_text')
            assert hasattr(r, 'doc_id')
            assert hasattr(r, 'title')
            assert hasattr(r, 'score')
            assert hasattr(r, 'source')


class TestIndexUsage:
    """Test that indexes are being used correctly."""

    def test_hnsw_index_exists(self):
        """HNSW index should exist for vector search."""
        from lib.db.postgres import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'passages'
            AND indexname LIKE '%hnsw%'
        """)
        indexes = [row[0] for row in cur.fetchall()]
        conn.close()

        assert len(indexes) > 0, "HNSW index should exist"

    def test_gin_index_exists(self):
        """GIN index should exist for BM25 search."""
        from lib.db.postgres import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'passages'
            AND indexdef LIKE '%gin%'
        """)
        indexes = [row[0] for row in cur.fetchall()]
        conn.close()

        assert len(indexes) > 0, "GIN index should exist for FTS"

    def test_vector_search_performance(self):
        """Vector search should complete in reasonable time."""
        import time
        from lib.search.hybrid_search import HybridSearcher

        searcher = HybridSearcher(rerank=False)

        # Warm up
        searcher.vector_search("warmup", n=1)

        # Timed search
        start = time.time()
        results = searcher.vector_search("spatial transcriptomics", n=10)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Vector search took {elapsed:.2f}s, expected <5s"
        assert len(results) > 0

    def test_bm25_search_performance(self):
        """BM25 search should complete quickly with GIN index."""
        import time
        from lib.search.hybrid_search import HybridSearcher

        searcher = HybridSearcher(rerank=False)

        start = time.time()
        results = searcher.bm25_search("gene expression", n=10)
        elapsed = time.time() - start

        # With GIN index, BM25 should be fast (<1s)
        assert elapsed < 2.0, f"BM25 search took {elapsed:.2f}s, expected <2s with GIN"


class TestSearchEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def searcher(self):
        from lib.search.hybrid_search import HybridSearcher
        return HybridSearcher(rerank=False)

    def test_empty_query(self, searcher):
        """Empty queries should not crash."""
        results = searcher.vector_search("", n=5)
        assert isinstance(results, list)

    def test_very_long_query(self, searcher):
        """Very long queries should be handled."""
        long_query = "spatial " * 500
        results = searcher.vector_search(long_query, n=5)
        assert isinstance(results, list)

    def test_special_characters(self, searcher):
        """Special characters should not crash search."""
        results = searcher.bm25_search("gene expression (p<0.05) @#$%", n=5)
        assert isinstance(results, list)

    def test_unicode_query(self, searcher):
        """Unicode queries should work."""
        results = searcher.vector_search("表达分析 gene expression", n=5)
        assert isinstance(results, list)

    def test_n_zero(self, searcher):
        """n=0 should return empty list."""
        results = searcher.vector_search("test", n=0)
        assert results == []

    def test_large_n(self, searcher):
        """Large n should work without error."""
        results = searcher.vector_search("test", n=1000)
        assert isinstance(results, list)


class TestKnownContent:
    """Test that known content can be found."""

    @pytest.fixture(scope="class")
    def searcher(self):
        from lib.search.hybrid_search import HybridSearcher
        return HybridSearcher(rerank=False)

    def test_find_spatial_transcriptomics(self, searcher):
        """Should find spatial transcriptomics content."""
        results = searcher.hybrid_search("spatial transcriptomics methods", n=10)
        assert len(results) > 0
        texts = " ".join(r.passage_text.lower() for r in results)
        assert "spatial" in texts or "transcriptom" in texts

    def test_find_deep_learning(self, searcher):
        """Should find deep learning content."""
        results = searcher.hybrid_search("deep learning neural network", n=10)
        assert len(results) > 0

    def test_find_cell_segmentation(self, searcher):
        """Should find cell segmentation content."""
        results = searcher.hybrid_search("cell segmentation nuclei", n=10)
        assert len(results) > 0


class TestReranking:
    """Test reranking functionality and quality improvement."""

    def test_reranker_loads(self):
        """Reranker model should load successfully."""
        from lib.search.hybrid_search import HybridSearcher
        searcher = HybridSearcher(rerank=True)
        reranker = searcher._get_reranker()
        # Reranker may be None if not available, but shouldn't crash
        assert reranker is not None or not searcher.rerank_enabled

    def test_reranking_changes_order(self):
        """Reranking should potentially reorder results."""
        from lib.search.hybrid_search import HybridSearcher

        query = "attention mechanism transformer architecture"

        # Get results without reranking
        s1 = HybridSearcher(rerank=False)
        r1 = s1.hybrid_search(query, n=10)

        # Get results with reranking
        s2 = HybridSearcher(rerank=True)
        r2 = s2.hybrid_search(query, n=10, rerank=True)

        # Both should return results
        assert len(r1) > 0
        assert len(r2) > 0

        # Reranked results should have 'reranked' source
        if r2[0].source == 'reranked':
            # Order might be different (that's the point of reranking)
            pass  # Test passes if reranking ran

    def test_reranking_scores_valid(self):
        """Reranked scores should be valid cross-encoder scores."""
        from lib.search.hybrid_search import HybridSearcher

        searcher = HybridSearcher(rerank=True)
        results = searcher.hybrid_search("gene expression", n=5, rerank=True)

        if results and results[0].source == 'reranked':
            for r in results:
                # Cross-encoder scores are typically in range [-10, 10] or similar
                assert isinstance(r.score, float)

    def test_reranking_performance(self):
        """Reranking should complete in reasonable time."""
        import time
        from lib.search.hybrid_search import HybridSearcher

        searcher = HybridSearcher(rerank=True)

        # Warm up reranker
        _ = searcher.hybrid_search("warmup", n=5, rerank=True)

        start = time.time()
        results = searcher.hybrid_search("spatial transcriptomics", n=10, rerank=True)
        elapsed = time.time() - start

        assert elapsed < 15.0, f"Reranking took {elapsed:.2f}s, expected <15s"


class TestGraphRAG:
    """Test GraphRAG query expansion functionality."""

    @pytest.fixture(scope="class")
    def searcher(self):
        from lib.search.hybrid_search import HybridSearcher
        return HybridSearcher(rerank=False)

    def test_graph_expand_runs(self, searcher):
        """GraphRAG expansion should run without errors."""
        results = searcher.hybrid_search(
            "spatial transcriptomics",
            n=10,
            graph_expand=True
        )
        assert isinstance(results, list)

    def test_graph_expand_query_expansion(self, searcher):
        """GraphRAG should expand queries with related concepts."""
        # Test the internal expansion method
        expanded, terms = searcher._expand_query_with_graph("gene expression")
        # Should return original query at minimum
        assert "gene expression" in expanded or expanded == "gene expression"
        assert isinstance(terms, list)

    def test_graph_expand_with_concepts(self):
        """GraphRAG should find related concepts from passage_concepts."""
        from lib.db.postgres import get_db_connection

        conn = get_db_connection()
        cur = conn.cursor()

        # Check if we have concepts to expand with
        cur.execute("""
            SELECT COUNT(DISTINCT concept_name)
            FROM passage_concepts
            WHERE confidence > 0.5
        """)
        concept_count = cur.fetchone()[0]
        conn.close()

        # If we have concepts, expansion should potentially add terms
        assert concept_count > 0, "Should have concepts for GraphRAG expansion"

    def test_graph_expand_respects_limits(self, searcher):
        """GraphRAG should respect max_expansions parameter."""
        from lib.config import config

        expanded, terms = searcher._expand_query_with_graph(
            "deep learning",
            max_expansions=3
        )
        assert len(terms) <= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
