#!/usr/bin/env python3
"""
Harvest papers from underrepresented polymathic fields.

Targets fields that could provide novel methods for spatial biology:
- Topological Data Analysis
- Sheaf Theory / Category Theory
- Game Theory (for cell competition)
- Control Theory (for gene regulation)
- Compressed Sensing (for sparse reconstruction)
- Tropical Geometry (for optimization)

Usage:
    python scripts/harvest_polymathic.py --dry-run           # Preview what would be found
    python scripts/harvest_polymathic.py --field tda         # Harvest TDA papers
    python scripts/harvest_polymathic.py --all               # Harvest all underrepresented fields
    python scripts/harvest_polymathic.py --all --auto-ingest # Find and ingest
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Polymathic field definitions with search queries
POLYMATHIC_FIELDS = {
    "tda": {
        "name": "Topological Data Analysis",
        "current_mentions": 69,
        "target": 200,
        "queries": [
            "persistent homology single cell",
            "topological data analysis biology",
            "persistent homology gene expression",
            "mapper algorithm transcriptomics",
            "Betti numbers tissue",
            "TDA spatial omics",
        ]
    },
    "sheaf": {
        "name": "Sheaf Theory / Category Theory",
        "current_mentions": 100,
        "target": 200,
        "queries": [
            "sheaf theory machine learning",
            "cellular sheaves graphs",
            "category theory biology",
            "sheaf neural networks",
            "compositional machine learning",
            "functorial data analysis",
        ]
    },
    "game_theory": {
        "name": "Game Theory",
        "current_mentions": 9,
        "target": 100,
        "queries": [
            "game theory tumor microenvironment",
            "evolutionary game theory cancer",
            "Nash equilibrium cell competition",
            "game theory immunology",
            "spatial games ecology",
            "evolutionary dynamics cells",
        ]
    },
    "control": {
        "name": "Control Theory",
        "current_mentions": 4,
        "target": 100,
        "queries": [
            "control theory gene regulation",
            "feedback control biological systems",
            "optimal control cell fate",
            "dynamical systems gene networks",
            "stability analysis transcription",
            "control theory synthetic biology",
        ]
    },
    "compressed_sensing": {
        "name": "Compressed Sensing",
        "current_mentions": 2,
        "target": 50,
        "queries": [
            "compressed sensing genomics",
            "sparse reconstruction single cell",
            "L1 minimization biology",
            "compressive sensing RNA-seq",
            "sparse recovery gene expression",
        ]
    },
    "tropical": {
        "name": "Tropical Geometry",
        "current_mentions": 17,
        "target": 50,
        "queries": [
            "tropical geometry biology",
            "tropical algebra optimization",
            "max-plus algebra systems biology",
            "tropical methods phylogenetics",
        ]
    },
    "info_geometry": {
        "name": "Information Geometry",
        "current_mentions": 0,  # Need to check
        "target": 50,
        "queries": [
            "information geometry machine learning",
            "Fisher information biology",
            "natural gradient neural networks",
            "statistical manifolds",
        ]
    },
    "renormalization": {
        "name": "Renormalization / Multiscale",
        "current_mentions": 0,
        "target": 50,
        "queries": [
            "renormalization group biology",
            "multiscale modeling spatial",
            "coarse graining cellular",
            "scale invariance gene expression",
        ]
    }
}


def run_discovery(query: str, limit: int = 20, auto_ingest: bool = False, dry_run: bool = False):
    """Run discover_papers.py for a query."""
    cmd = [
        sys.executable,
        "scripts/discover_papers.py",
        query,
        "--limit", str(limit),
    ]
    if auto_ingest and not dry_run:
        cmd.append("--auto-ingest")
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings: {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        print(f"  Timeout on query: {query}")
    except Exception as e:
        print(f"  Error: {e}")


def harvest_field(field_key: str, auto_ingest: bool = False, dry_run: bool = False, limit_per_query: int = 15):
    """Harvest papers for a specific polymathic field."""
    if field_key not in POLYMATHIC_FIELDS:
        print(f"Unknown field: {field_key}")
        print(f"Available: {', '.join(POLYMATHIC_FIELDS.keys())}")
        return

    field = POLYMATHIC_FIELDS[field_key]
    print(f"\n{'#'*60}")
    print(f"# Harvesting: {field['name']}")
    print(f"# Current mentions: {field['current_mentions']}, Target: {field['target']}")
    print(f"{'#'*60}")

    for query in field["queries"]:
        run_discovery(query, limit=limit_per_query, auto_ingest=auto_ingest, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(description="Harvest polymathic papers")
    parser.add_argument("--field", choices=list(POLYMATHIC_FIELDS.keys()),
                        help="Specific field to harvest")
    parser.add_argument("--all", action="store_true", help="Harvest all underrepresented fields")
    parser.add_argument("--auto-ingest", action="store_true", help="Auto-ingest found papers")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't download")
    parser.add_argument("--limit", type=int, default=15, help="Papers per query (default: 15)")
    parser.add_argument("--list", action="store_true", help="List available fields")

    args = parser.parse_args()

    if args.list:
        print("\nPolymathic Fields Available for Harvest:")
        print("-" * 60)
        for key, field in POLYMATHIC_FIELDS.items():
            gap = field["target"] - field["current_mentions"]
            print(f"  {key:20} | {field['name']:30} | Gap: {gap:+d}")
        return

    if args.all:
        # Prioritize by gap size
        fields_by_gap = sorted(
            POLYMATHIC_FIELDS.keys(),
            key=lambda k: POLYMATHIC_FIELDS[k]["target"] - POLYMATHIC_FIELDS[k]["current_mentions"],
            reverse=True
        )
        for field_key in fields_by_gap:
            harvest_field(field_key, auto_ingest=args.auto_ingest,
                         dry_run=args.dry_run, limit_per_query=args.limit)
    elif args.field:
        harvest_field(args.field, auto_ingest=args.auto_ingest,
                     dry_run=args.dry_run, limit_per_query=args.limit)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
