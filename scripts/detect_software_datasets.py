#!/usr/bin/env python3
"""
Fine-Grained NER: Software and Dataset Detection

Detects mentions of software tools and datasets in passages using:
1. Registry-based matching (known tools/datasets)
2. Pattern-based detection (GitHub URLs, common patterns)
3. Optional LLM-based extraction for complex cases

Usage:
    python scripts/detect_software_datasets.py --scan --limit 1000
    python scripts/detect_software_datasets.py --scan --llm-assist  # Use Gemini for ambiguous cases
    python scripts/detect_software_datasets.py --stats
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Pattern Definitions
# ============================================================================

# Common software tools (lowercase for matching)
SOFTWARE_PATTERNS = [
    # Python packages
    r'\b(scanpy|seurat|squidpy|anndata|cellpose|stardist)\b',
    r'\b(pytorch|tensorflow|keras|jax|numpy|pandas|scipy)\b',
    r'\b(scikit-learn|sklearn|xgboost|lightgbm)\b',
    r'\b(matplotlib|seaborn|plotly|bokeh)\b',
    r'\b(cell2location|tangram|stereoscope|rctd)\b',
    r'\b(spatialdata|napari|qupath|imagej|fiji)\b',
    r'\b(cellranger|spaceranger|loupe\s*browser)\b',

    # R packages
    r'\b(ggplot2|dplyr|tidyr|tidyverse)\b',
    r'\b(bioconductor|deseq2|edger)\b',

    # Deep learning
    r'\b(resnet|vgg|unet|transformer|bert|gpt)\b',
    r'\b(attention\s+mechanism|self-attention)\b',
]

# Dataset patterns
DATASET_PATTERNS = [
    # Spatial platforms
    r'\b(visium|xenium|merscope|cosmx|codex|stereo-seq)\b',
    r'\b(slide-seq|seq-scope|hdst)\b',

    # Genomics
    r'\b(tcga|gtex|hpa|geo|sra)\b',
    r'\b(encode|roadmap|fantom)\b',
    r'\b(gnomad|clinvar|cosmic)\b',

    # Imaging
    r'\b(imagenet|coco|cityscapes|ade20k)\b',
    r'\b(pannuke|monuseg|cpm\d+)\b',

    # Single-cell
    r'\b(tabula\s*(sapiens|muris))\b',
    r'\b(human\s*cell\s*atlas|hca)\b',
]

# GitHub URL pattern
GITHUB_PATTERN = r'github\.com/([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)'


# ============================================================================
# Detection Functions
# ============================================================================

def compile_patterns():
    """Compile regex patterns for efficiency."""
    software_re = re.compile('|'.join(SOFTWARE_PATTERNS), re.IGNORECASE)
    dataset_re = re.compile('|'.join(DATASET_PATTERNS), re.IGNORECASE)
    github_re = re.compile(GITHUB_PATTERN)
    return software_re, dataset_re, github_re


def load_registries(conn) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Load registries from database.

    Returns:
        Tuple of (software_lookup, dataset_lookup) where
        lookup = {alias.lower(): canonical_name}
    """
    cur = conn.cursor()

    # Software
    cur.execute("SELECT name, canonical_name, aliases FROM software_registry")
    software_lookup = {}
    for name, canonical, aliases in cur.fetchall():
        software_lookup[name.lower()] = canonical
        if aliases:
            for alias in aliases:
                software_lookup[alias.lower()] = canonical

    # Datasets
    cur.execute("SELECT name, canonical_name FROM dataset_registry")
    dataset_lookup = {}
    for name, canonical in cur.fetchall():
        dataset_lookup[name.lower()] = canonical

    return software_lookup, dataset_lookup


def detect_in_passage(
    passage_text: str,
    software_re,
    dataset_re,
    github_re,
    software_lookup: Dict[str, str],
    dataset_lookup: Dict[str, str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect software and dataset mentions in a passage.

    Returns:
        Tuple of (software_mentions, dataset_mentions)
    """
    text_lower = passage_text.lower()

    software_mentions = []
    dataset_mentions = []

    # Software pattern matching
    for match in software_re.finditer(passage_text):
        name = match.group().lower().strip()
        canonical = software_lookup.get(name, name.title())

        # Get context (50 chars before/after)
        start = max(0, match.start() - 50)
        end = min(len(passage_text), match.end() + 50)
        context = passage_text[start:end]

        software_mentions.append({
            'name': name,
            'canonical': canonical,
            'confidence': 0.9 if name in software_lookup else 0.7,
            'context': context
        })

    # GitHub URLs (high confidence software detection)
    for match in github_re.finditer(passage_text):
        org, repo = match.groups()
        full_name = f"{org}/{repo}"
        canonical = software_lookup.get(repo.lower(), repo)

        software_mentions.append({
            'name': repo.lower(),
            'canonical': canonical,
            'confidence': 0.95,
            'context': full_name
        })

    # Dataset pattern matching
    for match in dataset_re.finditer(passage_text):
        name = match.group().lower().strip()
        canonical = dataset_lookup.get(name, name.upper())

        start = max(0, match.start() - 50)
        end = min(len(passage_text), match.end() + 50)
        context = passage_text[start:end]

        dataset_mentions.append({
            'name': name,
            'canonical': canonical,
            'confidence': 0.9 if name in dataset_lookup else 0.7,
            'context': context
        })

    # Deduplicate by name
    seen_software = set()
    unique_software = []
    for m in software_mentions:
        if m['name'] not in seen_software:
            seen_software.add(m['name'])
            unique_software.append(m)

    seen_datasets = set()
    unique_datasets = []
    for m in dataset_mentions:
        if m['name'] not in seen_datasets:
            seen_datasets.add(m['name'])
            unique_datasets.append(m)

    return unique_software, unique_datasets


def store_mentions(
    conn,
    passage_id: str,
    software_mentions: List[Dict],
    dataset_mentions: List[Dict]
):
    """Store detected mentions in database."""
    cur = conn.cursor()

    for m in software_mentions:
        cur.execute("""
            INSERT INTO software_mentions (passage_id, software_name, canonical_name, confidence, context)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (passage_id, software_name) DO UPDATE SET
                canonical_name = EXCLUDED.canonical_name,
                confidence = EXCLUDED.confidence,
                context = EXCLUDED.context
        """, (passage_id, m['name'], m['canonical'], m['confidence'], m['context'][:500]))

    for m in dataset_mentions:
        cur.execute("""
            INSERT INTO dataset_mentions (passage_id, dataset_name, canonical_name, confidence, context)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (passage_id, dataset_name) DO UPDATE SET
                canonical_name = EXCLUDED.canonical_name,
                confidence = EXCLUDED.confidence,
                context = EXCLUDED.context
        """, (passage_id, m['name'], m['canonical'], m['confidence'], m['context'][:500]))

    conn.commit()


# ============================================================================
# Main Scan Logic
# ============================================================================

def scan_passages(
    conn,
    limit: int = 1000,
    skip_scanned: bool = True
) -> Dict:
    """
    Scan passages for software and dataset mentions.

    Returns:
        Summary dict with counts
    """
    cur = conn.cursor()

    # Get passages to scan
    if skip_scanned:
        cur.execute("""
            SELECT p.passage_id, p.passage_text
            FROM passages p
            WHERE p.passage_id NOT IN (
                SELECT DISTINCT passage_id FROM software_mentions
                UNION
                SELECT DISTINCT passage_id FROM dataset_mentions
            )
            AND LENGTH(p.passage_text) > 100
            AND p.is_superseded = FALSE
            LIMIT %s
        """, (limit,))
    else:
        cur.execute("""
            SELECT p.passage_id, p.passage_text
            FROM passages p
            WHERE LENGTH(p.passage_text) > 100
            AND p.is_superseded = FALSE
            LIMIT %s
        """, (limit,))

    passages = [(str(row[0]), row[1]) for row in cur.fetchall()]
    logger.info(f"Scanning {len(passages)} passages for software/dataset mentions...")

    # Load registries and compile patterns
    software_lookup, dataset_lookup = load_registries(conn)
    software_re, dataset_re, github_re = compile_patterns()

    # Counters
    total_software = 0
    total_datasets = 0
    software_counter = Counter()
    dataset_counter = Counter()

    for i, (passage_id, text) in enumerate(passages):
        software, datasets = detect_in_passage(
            text,
            software_re, dataset_re, github_re,
            software_lookup, dataset_lookup
        )

        if software or datasets:
            store_mentions(conn, passage_id, software, datasets)

            total_software += len(software)
            total_datasets += len(datasets)

            for m in software:
                software_counter[m['canonical']] += 1
            for m in datasets:
                dataset_counter[m['canonical']] += 1

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i+1}/{len(passages)} passages...")

    return {
        'passages_scanned': len(passages),
        'software_mentions': total_software,
        'dataset_mentions': total_datasets,
        'top_software': dict(software_counter.most_common(20)),
        'top_datasets': dict(dataset_counter.most_common(20))
    }


def get_stats(conn) -> Dict:
    """Get current detection statistics."""
    cur = conn.cursor()

    # Software stats
    cur.execute("SELECT COUNT(*) FROM software_mentions")
    software_count = cur.fetchone()[0]

    cur.execute("""
        SELECT canonical_name, COUNT(*) as cnt
        FROM software_mentions
        GROUP BY canonical_name
        ORDER BY cnt DESC
        LIMIT 20
    """)
    top_software = dict(cur.fetchall())

    # Dataset stats
    cur.execute("SELECT COUNT(*) FROM dataset_mentions")
    dataset_count = cur.fetchone()[0]

    cur.execute("""
        SELECT canonical_name, COUNT(*) as cnt
        FROM dataset_mentions
        GROUP BY canonical_name
        ORDER BY cnt DESC
        LIMIT 20
    """)
    top_datasets = dict(cur.fetchall())

    # Registry stats
    cur.execute("SELECT COUNT(*) FROM software_registry")
    software_registry = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM dataset_registry")
    dataset_registry = cur.fetchone()[0]

    return {
        'software_mentions': software_count,
        'dataset_mentions': dataset_count,
        'software_registry_size': software_registry,
        'dataset_registry_size': dataset_registry,
        'top_software': top_software,
        'top_datasets': top_datasets
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Detect software and dataset mentions')
    parser.add_argument('--scan', action='store_true', help='Scan passages for mentions')
    parser.add_argument('--stats', action='store_true', help='Show detection statistics')
    parser.add_argument('--limit', type=int, default=1000, help='Max passages to scan')
    parser.add_argument('--rescan', action='store_true', help='Rescan already processed passages')
    parser.add_argument('--output', '-o', type=Path, help='Output JSON file')

    args = parser.parse_args()

    conn = psycopg2.connect(config.POSTGRES_DSN)

    if args.scan:
        results = scan_passages(conn, args.limit, skip_scanned=not args.rescan)

        print(f"\n{'='*60}")
        print("SCAN RESULTS")
        print(f"{'='*60}")
        print(f"Passages scanned: {results['passages_scanned']}")
        print(f"Software mentions found: {results['software_mentions']}")
        print(f"Dataset mentions found: {results['dataset_mentions']}")

        if results['top_software']:
            print("\nTop Software:")
            for name, count in list(results['top_software'].items())[:10]:
                print(f"  {name}: {count}")

        if results['top_datasets']:
            print("\nTop Datasets:")
            for name, count in list(results['top_datasets'].items())[:10]:
                print(f"  {name}: {count}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)

    elif args.stats:
        stats = get_stats(conn)

        print(f"\n{'='*60}")
        print("DETECTION STATISTICS")
        print(f"{'='*60}")
        print(f"Total software mentions: {stats['software_mentions']}")
        print(f"Total dataset mentions: {stats['dataset_mentions']}")
        print(f"Software registry size: {stats['software_registry_size']}")
        print(f"Dataset registry size: {stats['dataset_registry_size']}")

        if stats['top_software']:
            print("\nTop Software Detected:")
            for name, count in list(stats['top_software'].items())[:10]:
                print(f"  {name}: {count}")

        if stats['top_datasets']:
            print("\nTop Datasets Detected:")
            for name, count in list(stats['top_datasets'].items())[:10]:
                print(f"  {name}: {count}")

    else:
        parser.print_help()

    conn.close()


if __name__ == '__main__':
    main()
