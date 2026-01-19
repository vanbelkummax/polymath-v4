#!/usr/bin/env python3
"""
Metadata enrichment via CrossRef API.

Fetches missing year, authors, and venue data for documents with DOIs.
"""

import argparse
import re
import time
import sys
from pathlib import Path

import httpx
from tqdm import tqdm

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.db.postgres import get_connection


def fetch_crossref_metadata(doi: str, timeout: float = 10.0) -> dict | None:
    """Fetch metadata from CrossRef API for a given DOI."""
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": "Polymath/1.0 (mailto:max.vanbelkum@vanderbilt.edu)"
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                message = data.get("message", {})

                # Extract year
                year = None
                if "published-print" in message:
                    year = message["published-print"].get("date-parts", [[None]])[0][0]
                elif "published-online" in message:
                    year = message["published-online"].get("date-parts", [[None]])[0][0]
                elif "created" in message:
                    year = message["created"].get("date-parts", [[None]])[0][0]

                # Extract authors
                authors = []
                for author in message.get("author", []):
                    if "family" in author:
                        name = author.get("given", "") + " " + author["family"]
                        authors.append(name.strip())

                # Extract venue
                venue = None
                if "container-title" in message and message["container-title"]:
                    venue = message["container-title"][0]

                # Extract abstract
                abstract = message.get("abstract", None)
                if abstract:
                    abstract = re.sub(r'<[^>]+>', '', abstract)

                return {
                    "year": year,
                    "authors": authors if authors else None,
                    "venue": venue,
                    "abstract": abstract,
                }
            elif resp.status_code == 404:
                return None
            else:
                return None
    except Exception as e:
        print(f"  Error fetching {doi}: {e}")
        return None


def enrich_documents(limit: int = 100, dry_run: bool = False, delay: float = 0.1):
    """Enrich documents with missing metadata via CrossRef."""
    # Find documents with DOI but missing year
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT doc_id, doi, title, authors, year, venue, abstract
                FROM documents
                WHERE doi IS NOT NULL
                  AND (year IS NULL OR year = 0)
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()

    print(f"Found {len(rows)} documents with DOI but missing year")

    if not rows:
        print("Nothing to enrich!")
        return

    updated = 0
    failed = 0

    for row in tqdm(rows, desc="Enriching"):
        doc_id, doi, title, current_authors, current_year, current_venue, current_abstract = row

        # Clean DOI
        doi_clean = doi.strip()
        if doi_clean.startswith("https://doi.org/"):
            doi_clean = doi_clean[16:]
        elif doi_clean.startswith("http://doi.org/"):
            doi_clean = doi_clean[15:]
        elif doi_clean.startswith("doi:"):
            doi_clean = doi_clean[4:]

        metadata = fetch_crossref_metadata(doi_clean)

        if metadata is None:
            failed += 1
            continue

        # Build update
        updates = []
        params = []

        if metadata["year"] and not current_year:
            updates.append("year = %s")
            params.append(metadata["year"])

        if metadata["authors"] and not current_authors:
            updates.append("authors = %s")
            params.append(metadata["authors"])

        if metadata["venue"] and not current_venue:
            updates.append("venue = %s")
            params.append(metadata["venue"])

        if metadata["abstract"] and not current_abstract:
            updates.append("abstract = %s")
            params.append(metadata["abstract"])

        if not updates:
            continue

        if dry_run:
            print(f"  Would update {title[:50]}... with: year={metadata['year']}, venue={metadata['venue']}")
        else:
            params.append(doc_id)
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        UPDATE documents
                        SET {', '.join(updates)}
                        WHERE doc_id = %s
                    """, params)
                conn.commit()
            updated += 1

        time.sleep(delay)  # Rate limiting

    print(f"\nEnrichment complete:")
    print(f"  Updated: {updated}")
    print(f"  Failed/NotFound: {failed}")
    print(f"  Skipped (no new data): {len(rows) - updated - failed}")


def stats():
    """Show current metadata coverage."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(year) as has_year,
                    COUNT(authors) FILTER (WHERE array_length(authors, 1) > 0) as has_authors,
                    COUNT(venue) as has_venue,
                    COUNT(abstract) as has_abstract,
                    COUNT(doi) as has_doi,
                    COUNT(*) FILTER (WHERE doi IS NOT NULL AND year IS NULL) as enrichable
                FROM documents
            """)
            row = cur.fetchone()

    total, has_year, has_authors, has_venue, has_abstract, has_doi, enrichable = row

    print("Metadata Coverage:")
    print(f"  Total documents:   {total}")
    print(f"  Has year:          {has_year} ({100*has_year/total:.1f}%)")
    print(f"  Has authors:       {has_authors} ({100*has_authors/total:.1f}%)")
    print(f"  Has venue:         {has_venue} ({100*has_venue/total:.1f}%)")
    print(f"  Has abstract:      {has_abstract} ({100*has_abstract/total:.1f}%)")
    print(f"  Has DOI:           {has_doi} ({100*has_doi/total:.1f}%)")
    print(f"  Enrichable (DOI, no year): {enrichable}")


def main():
    parser = argparse.ArgumentParser(description="Enrich document metadata via CrossRef")
    parser.add_argument("--limit", type=int, default=100, help="Max documents to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated")
    parser.add_argument("--stats", action="store_true", help="Show metadata coverage stats")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between API calls (seconds)")

    args = parser.parse_args()

    if args.stats:
        stats()
    else:
        enrich_documents(limit=args.limit, dry_run=args.dry_run, delay=args.delay)


if __name__ == "__main__":
    main()
