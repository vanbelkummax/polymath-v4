#!/usr/bin/env python3
"""
Fill citation network gaps - find and ingest papers that are cited but missing.

The active_librarian identified 688 DOIs that are mentioned in our corpus but
not present as documents. This script prioritizes and ingests them.

Usage:
    python scripts/fill_citation_gaps.py --analyze           # Show gap analysis
    python scripts/fill_citation_gaps.py --top 50            # Get top 50 missing DOIs
    python scripts/fill_citation_gaps.py --fetch --limit 20  # Fetch and ingest top 20
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db.postgres import get_pool


def analyze_citation_gaps():
    """Analyze missing DOIs from citation network."""
    pool = get_pool()

    # Find DOIs mentioned in text but not in documents
    query = """
    WITH mentioned_dois AS (
        SELECT DISTINCT
            regexp_matches(passage_text, '10\\.\\d{4,}/[^\\s<>\"'']+', 'gi') as doi_match
        FROM passages
    ),
    extracted_dois AS (
        SELECT LOWER(doi_match[1]) as doi
        FROM mentioned_dois
    ),
    existing_dois AS (
        SELECT LOWER(doi) as doi FROM documents WHERE doi IS NOT NULL
    )
    SELECT e.doi, COUNT(*) as mention_count
    FROM extracted_dois e
    LEFT JOIN existing_dois x ON e.doi = x.doi
    WHERE x.doi IS NULL
    GROUP BY e.doi
    ORDER BY mention_count DESC
    LIMIT 200;
    """

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()

    return results


def categorize_dois(dois_with_counts):
    """Categorize DOIs by publisher/journal."""
    categories = Counter()
    categorized = {}

    for doi, count in dois_with_counts:
        # Extract publisher prefix
        if doi.startswith("10.1038/"):
            cat = "Nature"
        elif doi.startswith("10.1016/j.cell"):
            cat = "Cell"
        elif doi.startswith("10.1126/"):
            cat = "Science"
        elif doi.startswith("10.1101/"):
            cat = "bioRxiv/medRxiv"
        elif doi.startswith("10.48550/"):
            cat = "arXiv"
        elif doi.startswith("10.1016/"):
            cat = "Elsevier"
        elif doi.startswith("10.1093/"):
            cat = "Oxford"
        elif doi.startswith("10.1371/"):
            cat = "PLOS"
        elif doi.startswith("10.1186/"):
            cat = "BMC"
        elif doi.startswith("10.3389/"):
            cat = "Frontiers"
        else:
            cat = "Other"

        categories[cat] += count
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append((doi, count))

    return categories, categorized


def fetch_and_ingest(dois: list, dry_run: bool = False):
    """Fetch papers by DOI and ingest them."""
    import subprocess

    for doi, count in dois:
        print(f"\n{'='*60}")
        print(f"DOI: {doi} (cited {count}x)")
        print(f"{'='*60}")

        if dry_run:
            print("  [DRY RUN] Would fetch and ingest")
            continue

        # Try to get PDF via unpaywall or other sources
        try:
            # Use discover_papers with DOI search
            cmd = [
                sys.executable,
                "scripts/discover_papers.py",
                f"doi:{doi}",
                "--limit", "1",
                "--auto-ingest"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(result.stdout[:500] if result.stdout else "No output")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fill citation network gaps")
    parser.add_argument("--analyze", action="store_true", help="Show gap analysis")
    parser.add_argument("--top", type=int, default=50, help="Show top N missing DOIs")
    parser.add_argument("--fetch", action="store_true", help="Fetch and ingest missing papers")
    parser.add_argument("--limit", type=int, default=20, help="Limit papers to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--category", help="Filter by category (Nature, Cell, Science, etc.)")

    args = parser.parse_args()

    print("Analyzing citation gaps...")
    gaps = analyze_citation_gaps()

    if not gaps:
        print("No citation gaps found!")
        return

    categories, categorized = categorize_dois(gaps)

    if args.analyze:
        print("\n" + "="*60)
        print("CITATION GAP ANALYSIS")
        print("="*60)
        print(f"\nTotal missing DOIs: {len(gaps)}")
        print(f"Total mentions: {sum(c for _, c in gaps)}")
        print("\nBy Publisher/Source:")
        for cat, count in categories.most_common():
            print(f"  {cat:20} {count:5} mentions")

        print("\nTop 20 Most-Cited Missing Papers:")
        for doi, count in gaps[:20]:
            print(f"  {count:4}x  {doi}")

    elif args.fetch:
        # Filter by category if specified
        if args.category:
            if args.category in categorized:
                target_dois = categorized[args.category][:args.limit]
            else:
                print(f"Unknown category: {args.category}")
                print(f"Available: {', '.join(categorized.keys())}")
                return
        else:
            target_dois = gaps[:args.limit]

        print(f"\nFetching {len(target_dois)} papers...")
        fetch_and_ingest(target_dois, dry_run=args.dry_run)

    else:
        print(f"\nTop {args.top} Missing DOIs:")
        for doi, count in gaps[:args.top]:
            print(f"  {count:4}x  {doi}")


if __name__ == "__main__":
    main()
