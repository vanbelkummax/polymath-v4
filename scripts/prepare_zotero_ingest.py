#!/usr/bin/env python3
"""
Prepare Zotero CSV for Polymath v4 Ingestion

Deduplicates entries and maps Windows paths to Linux paths.

Usage:
    python scripts/prepare_zotero_ingest.py '/mnt/c/Users/User/Downloads/My Library.csv'
    python scripts/prepare_zotero_ingest.py input.csv --output prepared.csv
"""

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Path mapping from Windows Zotero to Linux
PATH_MAPPINGS = [
    (r"C:\Users\User\Zotero\storage", "/mnt/c/Users/User/Zotero/storage"),
    (r"C:/Users/User/Zotero/storage", "/mnt/c/Users/User/Zotero/storage"),
]


def normalize_title(title: str) -> str:
    """Normalize title for deduplication."""
    return re.sub(r'[^a-z0-9]', '', title.lower())


def get_title_hash(title: str) -> str:
    """Generate hash for title deduplication."""
    normalized = normalize_title(title)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def map_windows_path(windows_path: str) -> Optional[str]:
    """Convert Windows path to Linux path."""
    if not windows_path:
        return None

    path = windows_path.strip()
    for win_prefix, linux_prefix in PATH_MAPPINGS:
        if path.startswith(win_prefix):
            return path.replace(win_prefix, linux_prefix).replace('\\', '/')

    # If no mapping found, try basic conversion
    if path.startswith('C:'):
        return '/mnt/c' + path[2:].replace('\\', '/')

    return path


def find_pdf_path(file_attachments: str) -> Optional[str]:
    """Extract PDF path from Zotero file attachments field."""
    if not file_attachments:
        return None

    # Zotero can have multiple attachments separated by semicolons
    for attachment in file_attachments.split(';'):
        attachment = attachment.strip()
        if attachment.lower().endswith('.pdf'):
            return map_windows_path(attachment)

    return None


def deduplicate_csv(input_path: str, output_path: str):
    """
    Deduplicate Zotero CSV with strict rules:
    1. DOI match = duplicate
    2. Title hash + year = duplicate (only if both match)
    """

    seen_dois = set()
    seen_title_year = set()
    duplicates = defaultdict(list)

    entries = []

    print(f"Reading: {input_path}")

    with open(input_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            doi = row.get('DOI', '').strip().lower()
            title = row.get('Title', '').strip()
            year = row.get('Publication Year', '').strip()

            # Check for DOI duplicate
            if doi:
                if doi in seen_dois:
                    duplicates['doi'].append((doi, title[:50]))
                    continue
                seen_dois.add(doi)

            # Check for title+year duplicate (only if no DOI)
            if not doi and title and year:
                title_hash = get_title_hash(title)
                key = f"{title_hash}:{year}"
                if key in seen_title_year:
                    duplicates['title_year'].append((title[:50], year))
                    continue
                seen_title_year.add(key)

            # Map file path
            pdf_path = find_pdf_path(row.get('File Attachments', ''))
            row['_pdf_path_linux'] = pdf_path or ''
            row['_title_hash'] = get_title_hash(title) if title else ''

            entries.append(row)

    # Write deduplicated CSV
    print(f"Writing: {output_path}")

    # Add our computed fields to fieldnames
    output_fieldnames = list(fieldnames) + ['_pdf_path_linux', '_title_hash']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(entries)

    # Report
    print(f"\n=== Deduplication Report ===")
    print(f"Input entries: {len(entries) + sum(len(v) for v in duplicates.values())}")
    print(f"Output entries: {len(entries)}")
    print(f"DOI duplicates removed: {len(duplicates['doi'])}")
    print(f"Title+year duplicates removed: {len(duplicates['title_year'])}")

    # Count entries with valid PDF paths
    with_pdf = sum(1 for e in entries if e['_pdf_path_linux'])
    print(f"\nEntries with PDF paths: {with_pdf} ({100*with_pdf/len(entries):.1f}%)")

    # Sample path mappings for verification
    print(f"\n=== Sample Path Mappings ===")
    samples = [e for e in entries if e['_pdf_path_linux']][:3]
    for s in samples:
        print(f"  {s.get('File Attachments', '')[:50]}...")
        print(f"  → {s['_pdf_path_linux'][:60]}...")
        print()

    return len(entries)


def main():
    parser = argparse.ArgumentParser(description='Prepare Zotero CSV for ingestion')
    parser.add_argument('input', help='Input Zotero CSV file')
    parser.add_argument('--output', '-o', help='Output prepared CSV file')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Default output path
    if args.output:
        output_path = args.output
    else:
        output_path = '/home/user/work/polymax/ingest_staging/zotero_prepared.csv'

    count = deduplicate_csv(str(input_path), output_path)
    print(f"\nReady for ingestion: {count} entries in {output_path}")


if __name__ == '__main__':
    main()
