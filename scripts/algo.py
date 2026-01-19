#!/usr/bin/env python3
"""
Quick algorithm lookup for hackathon/real-time use.

Usage:
    python scripts/algo.py "gradient descent"      # Search by name
    python scripts/algo.py --domain topology       # List by domain
    python scripts/algo.py --bridges               # Show polymathic bridges
    python scripts/algo.py --spatial               # Algorithms for spatial biology
    python scripts/algo.py --top 20                # Top algorithms by mentions
    python scripts/algo.py --ocr-issues            # Show OCR quality concerns
    python scripts/algo.py --for "cell clustering" # Find algorithms for a use case
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db.postgres import get_db_connection


def search_algorithm(conn, query):
    """Search for algorithms by name."""
    cur = conn.cursor()

    # Try exact match first
    cur.execute("""
        SELECT a.name, a.canonical_name, a.original_domain, a.category,
               a.mention_count, a.spatial_biology_uses, a.what_it_does,
               a.ocr_quality_flag
        FROM algorithms a
        WHERE a.name ILIKE %s
           OR a.canonical_name ILIKE %s
        ORDER BY a.mention_count DESC
        LIMIT 20
    """, (f'%{query}%', f'%{query}%'))

    results = cur.fetchall()

    if not results:
        print(f"No algorithms found matching '{query}'")
        return

    print(f"\n{'='*60}")
    print(f"ALGORITHMS MATCHING: {query}")
    print(f"{'='*60}\n")

    for name, canonical, domain, category, mentions, spatial_uses, desc, ocr_flag in results:
        print(f"📊 {canonical or name}")
        print(f"   Domain: {domain or 'unclassified'} | Category: {category or 'general'}")
        print(f"   Mentions: {mentions}")

        if spatial_uses:
            print(f"   Spatial uses: {', '.join(spatial_uses[:3])}")

        if desc:
            print(f"   Description: {desc[:100]}...")

        if ocr_flag == 'suspect':
            print(f"   ⚠️  OCR quality may need review")

        # Get linked papers
        cur.execute("""
            SELECT d.title, d.year
            FROM algorithm_papers ap
            JOIN documents d ON ap.doc_id = d.doc_id
            WHERE ap.algo_id = (SELECT algo_id FROM algorithms WHERE name = %s)
            ORDER BY d.year DESC
            LIMIT 3
        """, (name,))
        papers = cur.fetchall()
        if papers:
            print(f"   Papers: {papers[0][0][:50]}... ({papers[0][1]})")

        # Get linked repos
        cur.execute("""
            SELECT r.name, r.repo_url
            FROM algorithm_repos ar
            JOIN repositories r ON ar.repo_id = r.repo_id
            WHERE ar.algo_id = (SELECT algo_id FROM algorithms WHERE name = %s)
            LIMIT 2
        """, (name,))
        repos = cur.fetchall()
        if repos:
            print(f"   Code: {repos[0][1]}")

        print()


def list_by_domain(conn, domain):
    """List algorithms by domain."""
    cur = conn.cursor()

    cur.execute("""
        SELECT name, canonical_name, category, mention_count, spatial_biology_uses
        FROM algorithms
        WHERE original_domain = %s
        ORDER BY mention_count DESC
        LIMIT 30
    """, (domain,))

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print(f"ALGORITHMS IN DOMAIN: {domain.upper()}")
    print(f"{'='*60}\n")

    for name, canonical, category, mentions, spatial in results:
        spatial_str = f" → {spatial[0]}" if spatial else ""
        print(f"  {canonical or name} ({category}) [{mentions} mentions]{spatial_str}")


def show_polymathic_bridges(conn, limit=30):
    """Show polymathic cross-domain bridges."""
    cur = conn.cursor()

    cur.execute("""
        SELECT a.name, ab.source_domain, ab.target_domain, ab.polymathic_score
        FROM algorithm_bridges ab
        JOIN algorithms a ON ab.algo_id = a.algo_id
        ORDER BY ab.polymathic_score DESC
        LIMIT %s
    """, (limit,))

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print("POLYMATHIC BRIDGES (Cross-Domain Algorithm Transfers)")
    print(f"{'='*60}\n")

    for name, source, target, score in results:
        print(f"  🔗 {name}")
        print(f"     {source} → {target} (score: {score:.2f})")
        print()


def show_spatial_algorithms(conn, limit=30):
    """Show algorithms with spatial biology applications."""
    cur = conn.cursor()

    cur.execute("""
        SELECT name, original_domain, spatial_biology_uses, mention_count
        FROM algorithms
        WHERE spatial_biology_uses IS NOT NULL
          AND array_length(spatial_biology_uses, 1) > 0
        ORDER BY mention_count DESC
        LIMIT %s
    """, (limit,))

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print("ALGORITHMS FOR SPATIAL BIOLOGY")
    print(f"{'='*60}\n")

    for name, domain, uses, mentions in results:
        print(f"  📍 {name} ({domain})")
        for use in uses[:3]:
            print(f"     → {use}")
        print()


def show_top_algorithms(conn, limit=20):
    """Show top algorithms by mention count."""
    cur = conn.cursor()

    cur.execute("""
        SELECT name, original_domain, category, mention_count,
               (SELECT COUNT(*) FROM algorithm_papers ap WHERE ap.algo_id = a.algo_id) as paper_count,
               (SELECT COUNT(*) FROM algorithm_repos ar WHERE ar.algo_id = a.algo_id) as repo_count
        FROM algorithms a
        ORDER BY mention_count DESC
        LIMIT %s
    """, (limit,))

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print("TOP ALGORITHMS BY MENTIONS")
    print(f"{'='*60}\n")

    print(f"{'Algorithm':<35} {'Domain':<20} {'Mentions':>8} {'Papers':>7} {'Repos':>6}")
    print("-" * 80)

    for name, domain, cat, mentions, papers, repos in results:
        display_name = name[:33] + '..' if len(name) > 35 else name
        display_domain = (domain or 'unclassified')[:18]
        print(f"{display_name:<35} {display_domain:<20} {mentions:>8} {papers:>7} {repos:>6}")


def show_ocr_issues(conn):
    """Show algorithms with OCR quality concerns."""
    cur = conn.cursor()

    cur.execute("""
        SELECT name, original_domain, ocr_quality_notes, mention_count
        FROM algorithms
        WHERE ocr_quality_flag = 'suspect'
        ORDER BY mention_count DESC
        LIMIT 30
    """)

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print("ALGORITHMS WITH OCR QUALITY CONCERNS")
    print("(Math extraction may need manual review)")
    print(f"{'='*60}\n")

    if not results:
        print("  ✓ No OCR quality concerns flagged")
        return

    for name, domain, notes, mentions in results:
        print(f"  ⚠️  {name} ({domain})")
        print(f"     Mentions: {mentions}")
        if notes:
            print(f"     Issues: {notes[:100]}")
        print()


def find_for_usecase(conn, usecase):
    """Find algorithms suitable for a use case."""
    cur = conn.cursor()

    # Search in spatial_biology_uses and descriptions
    cur.execute("""
        SELECT name, original_domain, spatial_biology_uses, what_it_does, mention_count
        FROM algorithms
        WHERE %s = ANY(spatial_biology_uses)
           OR what_it_does ILIKE %s
           OR name ILIKE %s
        ORDER BY mention_count DESC
        LIMIT 20
    """, (usecase, f'%{usecase}%', f'%{usecase}%'))

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print(f"ALGORITHMS FOR: {usecase}")
    print(f"{'='*60}\n")

    if not results:
        print(f"  No algorithms found for '{usecase}'")
        print("  Try: cell clustering, gene imputation, spatial alignment, deconvolution")
        return

    for name, domain, spatial, desc, mentions in results:
        print(f"  🎯 {name} ({domain})")
        if spatial and usecase in spatial:
            print(f"     ✓ Direct application")
        if desc:
            print(f"     {desc[:80]}...")
        print()


def list_domains(conn):
    """List available domains."""
    cur = conn.cursor()

    cur.execute("""
        SELECT domain_name, description, is_polymathic_source, spatial_relevance
        FROM algorithm_domains
        ORDER BY is_polymathic_source DESC, domain_name
    """)

    results = cur.fetchall()

    print(f"\n{'='*60}")
    print("ALGORITHM DOMAINS")
    print(f"{'='*60}\n")

    for name, desc, is_poly, relevance in results:
        poly_marker = "🌟" if is_poly else "  "
        print(f"{poly_marker} {name:<25} [{relevance}]")
        if desc:
            print(f"    {desc[:60]}")
    print()
    print("🌟 = Rich source for polymathic transfers")


def main():
    parser = argparse.ArgumentParser(description="Quick algorithm lookup")
    parser.add_argument('query', nargs='?', help='Algorithm name to search')
    parser.add_argument('--domain', help='List algorithms by domain')
    parser.add_argument('--bridges', action='store_true', help='Show polymathic bridges')
    parser.add_argument('--spatial', action='store_true', help='Algorithms for spatial biology')
    parser.add_argument('--top', type=int, metavar='N', help='Show top N algorithms')
    parser.add_argument('--ocr-issues', action='store_true', help='Show OCR quality concerns')
    parser.add_argument('--for', dest='usecase', help='Find algorithms for a use case')
    parser.add_argument('--domains', action='store_true', help='List available domains')

    args = parser.parse_args()

    if not any([args.query, args.domain, args.bridges, args.spatial,
                args.top, args.ocr_issues, args.usecase, args.domains]):
        parser.print_help()
        print("\nExamples:")
        print("  algo.py 'gradient descent'")
        print("  algo.py --domain topology")
        print("  algo.py --bridges")
        print("  algo.py --for 'cell clustering'")
        return

    conn = get_db_connection()

    try:
        if args.query:
            search_algorithm(conn, args.query)
        elif args.domain:
            list_by_domain(conn, args.domain)
        elif args.bridges:
            show_polymathic_bridges(conn)
        elif args.spatial:
            show_spatial_algorithms(conn)
        elif args.top:
            show_top_algorithms(conn, args.top)
        elif args.ocr_issues:
            show_ocr_issues(conn)
        elif args.usecase:
            find_for_usecase(conn, args.usecase)
        elif args.domains:
            list_domains(conn)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
