#!/usr/bin/env python3
"""
Quick query interface for hackathon - minimal syntax.

Usage:
    python scripts/q.py "your query"
    python scripts/q.py "your query" -n 20
    python scripts/q.py "your query" --fast  # Skip reranking
    python scripts/q.py "your query" --repos  # Search repos instead
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger().setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(description="Quick search")
    parser.add_argument("query", nargs="+", help="Search query")
    parser.add_argument("-n", type=int, default=10, help="Number of results")
    parser.add_argument("--fast", action="store_true", help="Skip reranking")
    parser.add_argument("--repos", action="store_true", help="Search repos")
    parser.add_argument("--code", action="store_true", help="Find code/repos for papers")
    args = parser.parse_args()

    query = " ".join(args.query)

    if args.repos:
        # Search repo passages
        from lib.db.postgres import get_pool
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT r.name, r.stars, r.repo_url, LEFT(rp.passage_text, 200)
                    FROM repo_passages rp
                    JOIN repositories r ON rp.repo_id = r.repo_id
                    WHERE rp.passage_text ILIKE %s
                    ORDER BY r.stars DESC NULLS LAST
                    LIMIT %s
                """, (f"%{query}%", args.n))
                results = cur.fetchall()

        print(f"\n=== Repos matching '{query}' ===\n")
        for name, stars, url, text in results:
            stars_str = f"⭐{stars}" if stars else ""
            print(f"{name} {stars_str}")
            print(f"  {url}")
            print(f"  {text[:150]}...")
            print()

    elif args.code:
        # Find code for papers matching query via semantic search + repo links
        from lib.search.hybrid_search import HybridSearcher
        from lib.db.postgres import get_pool

        # First: semantic search for relevant papers
        searcher = HybridSearcher(rerank=not args.fast)
        paper_results = searcher.hybrid_search(query, n=args.n * 3)

        # Get unique doc_ids from search results
        doc_ids = list(set(r.doc_id for r in paper_results if hasattr(r, 'doc_id')))

        if not doc_ids:
            print(f"\n=== Code for papers matching '{query}' ===\n")
            print("No matching papers found.")
            return

        # Second: find repos linked to these papers
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Try paper_repo_links first (has repo_id FK)
                cur.execute("""
                    SELECT DISTINCT d.title, d.year, r.name, r.stars, r.repo_url, r.description
                    FROM documents d
                    JOIN paper_repo_links prl ON d.doc_id = prl.doc_id
                    JOIN repositories r ON prl.repo_id = r.repo_id
                    WHERE d.doc_id = ANY(%s::uuid[])
                    ORDER BY r.stars DESC NULLS LAST
                    LIMIT %s
                """, (doc_ids, args.n))
                results = cur.fetchall()

                # Fallback to paper_repos if no results
                if not results:
                    cur.execute("""
                        SELECT DISTINCT d.title, d.year, r.name, r.stars, r.repo_url, r.description
                        FROM documents d
                        JOIN paper_repos pr ON d.doc_id = pr.doc_id
                        JOIN repositories r ON LOWER(pr.repo_url) = LOWER(r.repo_url)
                        WHERE d.doc_id = ANY(%s::uuid[])
                        ORDER BY r.stars DESC NULLS LAST
                        LIMIT %s
                    """, (doc_ids, args.n))
                    results = cur.fetchall()

        print(f"\n=== Code for papers matching '{query}' ===\n")
        if not results:
            print("No repos linked to matching papers. Try running asset detection.")
        for title, year, name, stars, url, desc in results:
            stars_str = f"⭐{stars}" if stars else ""
            print(f"📄 {title[:60]}... ({year or 'n.d.'})")
            print(f"   💻 {name} {stars_str}")
            print(f"   {url}")
            if desc:
                print(f"   {desc[:100]}...")
            print()

    else:
        # Standard paper search
        from lib.search.hybrid_search import HybridSearcher
        searcher = HybridSearcher(rerank=not args.fast)
        results = searcher.hybrid_search(query, n=args.n)

        print(f"\n=== Results for '{query}' ===\n")
        for i, r in enumerate(results, 1):
            title = r.title[:70] if len(r.title) > 70 else r.title
            year = getattr(r, 'year', 'n.d.')
            print(f"{i}. {title}... ({year})")
            print(f"   {r.passage_text[:150]}...")
            print()


if __name__ == "__main__":
    main()
