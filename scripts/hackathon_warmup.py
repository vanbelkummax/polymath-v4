#!/usr/bin/env python3
"""
Hackathon Warmup Script - Run at start of hackathon session.

Pre-loads all models and caches common queries for instant access.

Usage:
    python scripts/hackathon_warmup.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    print("=" * 60)
    print("HACKATHON WARMUP - Loading models and caching queries")
    print("=" * 60)

    total_start = time.time()

    # 1. Warm up search with reranker
    print("\n[1/4] Loading embedding model + reranker...")
    start = time.time()
    from lib.search.hybrid_search import warmup
    searcher = warmup(rerank=True)
    print(f"  ✓ Done in {time.time() - start:.1f}s")

    # 2. Pre-cache common hackathon queries
    print("\n[2/4] Pre-caching common queries...")
    start = time.time()

    common_queries = [
        "spatial transcriptomics gene expression prediction",
        "histology image deep learning",
        "vision transformer pathology",
        "graph neural network spatial",
        "multimodal fusion contrastive",
        "cell segmentation nuclei",
        "attention mechanism spatial context",
        "optimal transport alignment",
        "foundation model pathology",
        "benchmark evaluation metrics",
    ]

    cache = {}
    for q in common_queries:
        results = searcher.hybrid_search(q, n=15)
        cache[q] = results
        print(f"  ✓ Cached: {q[:40]}... ({len(results)} results)")

    print(f"  Total cache time: {time.time() - start:.1f}s")

    # 3. Test database connection
    print("\n[3/4] Testing database connection...")
    start = time.time()
    from lib.db.postgres import get_pool
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            doc_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM repositories")
            repo_count = cur.fetchone()[0]
    print(f"  ✓ Connected: {doc_count} papers, {repo_count} repos")
    print(f"  Done in {time.time() - start:.1f}s")

    # 4. Quick test query
    print("\n[4/4] Running test query...")
    start = time.time()
    test_results = searcher.hybrid_search("spatial multimodal", n=5)
    print(f"  ✓ Query returned {len(test_results)} results in {time.time() - start:.2f}s")

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"WARMUP COMPLETE - Total time: {total_time:.1f}s")
    print("=" * 60)
    print("\nSystem ready for hackathon. Expected query time: ~2-3s")
    print("\nQuick search command:")
    print('  python -c "from lib.search.hybrid_search import search; print(search(\'YOUR_QUERY\', n=5))"')

    return searcher, cache


if __name__ == "__main__":
    main()
