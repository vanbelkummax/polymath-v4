#!/usr/bin/env python3
"""
Build and populate the Algorithm Registry from existing concept data.

This script:
1. Extracts algorithm-like concepts from passage_concepts
2. Classifies them by domain using heuristics + LLM
3. Links to source papers and repositories
4. Flags OCR quality concerns for math-heavy content
5. Identifies polymathic bridge opportunities

Usage:
    python scripts/build_algorithm_registry.py --extract     # Extract from concepts
    python scripts/build_algorithm_registry.py --classify    # LLM classification
    python scripts/build_algorithm_registry.py --link        # Link to papers/repos
    python scripts/build_algorithm_registry.py --bridges     # Find cross-domain bridges
    python scripts/build_algorithm_registry.py --ocr-audit   # Flag OCR quality issues
    python scripts/build_algorithm_registry.py --stats       # Show registry stats
    python scripts/build_algorithm_registry.py --full        # Run all steps
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db.postgres import get_db_connection

# Domain classification heuristics
DOMAIN_PATTERNS = {
    'topology': [
        r'persistent.?homology', r'betti', r'topological', r'simplicial',
        r'mapper', r'filtration', r'homology', r'cohomology', r'homotopy'
    ],
    'optimal_transport': [
        r'wasserstein', r'kantorovich', r'sinkhorn', r'earth.?mover',
        r'optimal.?transport', r'transport.?map'
    ],
    'linear_algebra': [
        r'singular.?value', r'eigenvalue', r'eigenvector', r'matrix',
        r'decomposition', r'factorization', r'svd', r'pca', r'nmf'
    ],
    'optimization': [
        r'gradient.?descent', r'adam', r'sgd', r'convex', r'lagrangian',
        r'newton', r'quasi.?newton', r'bfgs', r'conjugate.?gradient'
    ],
    'graph_theory': [
        r'louvain', r'leiden', r'pagerank', r'graph.?cut', r'spectral.?cluster',
        r'community.?detect', r'shortest.?path', r'minimum.?spanning'
    ],
    'signal_processing': [
        r'fourier', r'wavelet', r'fft', r'dct', r'filter', r'convolution',
        r'spectral', r'frequency'
    ],
    'statistics': [
        r'bayesian', r'monte.?carlo', r'mcmc', r'gibbs', r'metropolis',
        r'bootstrap', r'permutation.?test', r'hypothesis.?test'
    ],
    'machine_learning': [
        r'random.?forest', r'xgboost', r'gradient.?boost', r'svm',
        r'k.?means', r'dbscan', r'hierarchical.?cluster'
    ],
    'deep_learning': [
        r'backpropagation', r'attention', r'transformer', r'lstm', r'gru',
        r'resnet', r'unet', r'vae', r'gan', r'autoencoder'
    ],
    'control_theory': [
        r'kalman', r'pid', r'lqr', r'mpc', r'feedback.?control',
        r'state.?space', r'observability', r'controllability'
    ],
    'game_theory': [
        r'nash', r'equilibrium', r'pareto', r'minimax', r'stackelberg',
        r'evolutionary.?game', r'mechanism.?design'
    ],
    'information_theory': [
        r'entropy', r'mutual.?information', r'kl.?divergence', r'rate.?distortion',
        r'channel.?capacity', r'information.?gain'
    ],
    'compressed_sensing': [
        r'lasso', r'compressed.?sensing', r'sparse.?reconstruction',
        r'l1.?minimization', r'basis.?pursuit', r'restricted.?isometry'
    ],
    'category_theory': [
        r'functor', r'monad', r'sheaf', r'presheaf', r'natural.?transformation',
        r'adjoint', r'colimit', r'pullback'
    ],
}

# Algorithm category patterns
CATEGORY_PATTERNS = {
    'clustering': [r'cluster', r'k.?means', r'dbscan', r'hierarchical', r'louvain', r'leiden'],
    'decomposition': [r'decomposition', r'factorization', r'svd', r'nmf', r'pca', r'ica'],
    'transform': [r'transform', r'fourier', r'wavelet', r'laplace', r'z.?transform'],
    'optimization': [r'optimization', r'descent', r'minimize', r'maximize', r'convex'],
    'sampling': [r'sampling', r'monte.?carlo', r'mcmc', r'gibbs', r'metropolis'],
    'regression': [r'regression', r'linear', r'logistic', r'ridge', r'lasso'],
    'classification': [r'classifier', r'svm', r'random.?forest', r'decision.?tree'],
    'graph': [r'graph', r'network', r'shortest.?path', r'spanning.?tree', r'flow'],
    'neural': [r'neural', r'deep', r'attention', r'transformer', r'convolution'],
    'probabilistic': [r'probabilistic', r'bayesian', r'belief', r'inference'],
}

# OCR quality indicators (patterns that suggest math extraction issues)
OCR_SUSPECT_PATTERNS = [
    r'[^\x00-\x7F]{5,}',  # Long non-ASCII sequences
    r'\s{3,}',             # Excessive whitespace
    r'[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+',  # Fragmented numbers (from tables)
    r'^[A-Z][a-z]?\s*$',   # Single letters on lines (from equations)
    r'\([a-z]\)\s*$',      # Orphan equation labels
]


def extract_algorithms(conn, min_mentions=5):
    """Extract algorithm candidates from passage_concepts."""
    print("Extracting algorithm candidates from concepts...")

    cur = conn.cursor()

    # Get algorithm-like concepts
    cur.execute("""
        SELECT concept_name, concept_type, COUNT(*) as mentions,
               array_agg(DISTINCT passage_id) as passage_ids
        FROM passage_concepts
        WHERE concept_type IN ('algorithm', 'method', 'technique', 'math_object')
          AND LENGTH(concept_name) > 3
        GROUP BY concept_name, concept_type
        HAVING COUNT(*) >= %s
        ORDER BY COUNT(*) DESC
    """, (min_mentions,))

    candidates = cur.fetchall()
    print(f"Found {len(candidates)} algorithm candidates with >= {min_mentions} mentions")

    inserted = 0
    for name, ctype, mentions, passage_ids in candidates:
        # Skip generic terms
        if name.lower() in ['method', 'algorithm', 'approach', 'technique', 'analysis', 'model']:
            continue

        # Classify domain
        domain = classify_domain(name)
        category = classify_category(name)

        # Check if already exists
        cur.execute("SELECT algo_id FROM algorithms WHERE name = %s", (name,))
        if cur.fetchone():
            continue

        # Insert algorithm
        cur.execute("""
            INSERT INTO algorithms (name, canonical_name, original_domain, category, mention_count)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET mention_count = EXCLUDED.mention_count
            RETURNING algo_id
        """, (name, normalize_name(name), domain, category, mentions))

        algo_id = cur.fetchone()[0]
        inserted += 1

        # Link to papers via passages
        cur.execute("""
            INSERT INTO algorithm_papers (algo_id, doc_id, passage_ids)
            SELECT %s, p.doc_id, array_agg(p.passage_id)
            FROM passages p
            WHERE p.passage_id = ANY(%s::uuid[])
            GROUP BY p.doc_id
            ON CONFLICT (algo_id, doc_id) DO UPDATE
            SET passage_ids = EXCLUDED.passage_ids
        """, (algo_id, passage_ids))

    conn.commit()
    print(f"Inserted {inserted} new algorithms")
    return inserted


def classify_domain(name):
    """Classify algorithm domain using heuristics."""
    name_lower = name.lower().replace('_', ' ')

    for domain, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return domain

    return 'unclassified'


def classify_category(name):
    """Classify algorithm category."""
    name_lower = name.lower().replace('_', ' ')

    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return category

    return 'general'


def normalize_name(name):
    """Normalize algorithm name to canonical form."""
    # Convert underscores to spaces, title case
    normalized = name.replace('_', ' ').title()

    # Handle common abbreviations
    abbrev_map = {
        'Svd': 'SVD',
        'Pca': 'PCA',
        'Nmf': 'NMF',
        'Fft': 'FFT',
        'Mcmc': 'MCMC',
        'Svm': 'SVM',
        'Knn': 'KNN',
        'Lstm': 'LSTM',
        'Gru': 'GRU',
        'Gan': 'GAN',
        'Vae': 'VAE',
        'Rnn': 'RNN',
        'Cnn': 'CNN',
        'Mle': 'MLE',
        'Map': 'MAP',
        'Em ': 'EM ',
        'Lda': 'LDA',
    }

    for old, new in abbrev_map.items():
        normalized = normalized.replace(old, new)

    return normalized


def link_to_repos(conn):
    """Link algorithms to repository implementations."""
    print("Linking algorithms to repositories...")

    cur = conn.cursor()

    # Get all algorithms
    cur.execute("SELECT algo_id, name, canonical_name FROM algorithms")
    algorithms = cur.fetchall()

    linked = 0
    for algo_id, name, canonical_name in algorithms:
        # Search repo passages for algorithm mentions
        search_terms = [name.lower(), canonical_name.lower() if canonical_name else name.lower()]

        for term in search_terms:
            cur.execute("""
                INSERT INTO algorithm_repos (algo_id, repo_id, language)
                SELECT DISTINCT %s, rp.repo_id, r.language
                FROM repo_passages rp
                JOIN repositories r ON rp.repo_id = r.repo_id
                WHERE rp.passage_text ILIKE %s
                ON CONFLICT (algo_id, repo_id) DO NOTHING
            """, (algo_id, f'%{term}%'))

            if cur.rowcount > 0:
                linked += cur.rowcount

    conn.commit()
    print(f"Created {linked} algorithm-repo links")
    return linked


def find_polymathic_bridges(conn):
    """Identify cross-domain algorithm applications."""
    print("Finding polymathic bridges...")

    cur = conn.cursor()

    # Find algorithms used in unexpected domains
    cur.execute("""
        WITH algo_docs AS (
            SELECT a.algo_id, a.name, a.original_domain, d.doc_id, d.title
            FROM algorithms a
            JOIN algorithm_papers ap ON a.algo_id = ap.algo_id
            JOIN documents d ON ap.doc_id = d.doc_id
            WHERE a.original_domain IS NOT NULL
              AND a.original_domain != 'unclassified'
        ),
        doc_domains AS (
            -- Infer document domain from concepts
            SELECT doc_id,
                   CASE
                       WHEN title ILIKE '%spatial%' OR title ILIKE '%transcriptom%' THEN 'spatial_biology'
                       WHEN title ILIKE '%cancer%' OR title ILIKE '%tumor%' THEN 'oncology'
                       WHEN title ILIKE '%neural%' OR title ILIKE '%brain%' THEN 'neuroscience'
                       WHEN title ILIKE '%drug%' OR title ILIKE '%pharma%' THEN 'pharmacology'
                       WHEN title ILIKE '%cell%' AND title ILIKE '%type%' THEN 'cell_biology'
                       ELSE 'biology'
                   END as inferred_domain
            FROM documents
        )
        SELECT a.algo_id, a.name, a.original_domain, dd.inferred_domain, COUNT(*) as cross_uses
        FROM algo_docs a
        JOIN doc_domains dd ON a.doc_id = dd.doc_id
        WHERE a.original_domain NOT IN ('machine_learning', 'deep_learning', 'statistics', 'unclassified')
          AND dd.inferred_domain IN ('spatial_biology', 'cell_biology', 'oncology')
        GROUP BY a.algo_id, a.name, a.original_domain, dd.inferred_domain
        HAVING COUNT(*) >= 2
        ORDER BY COUNT(*) DESC
        LIMIT 100
    """)

    bridges = cur.fetchall()
    print(f"Found {len(bridges)} potential polymathic bridges")

    for algo_id, name, source_domain, target_domain, count in bridges:
        cur.execute("""
            INSERT INTO algorithm_bridges
            (algo_id, source_domain, target_domain, polymathic_score)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (algo_id, source_domain, target_domain, min(1.0, count / 10.0)))

    conn.commit()
    return len(bridges)


def audit_ocr_quality(conn):
    """Flag algorithms with potential OCR quality issues in source passages."""
    print("Auditing OCR quality for math-heavy content...")

    cur = conn.cursor()

    # Get passages for algorithms with math in their names
    cur.execute("""
        SELECT a.algo_id, a.name, p.passage_text, p.passage_id
        FROM algorithms a
        JOIN algorithm_papers ap ON a.algo_id = ap.algo_id
        JOIN passages p ON ap.doc_id = p.doc_id
        WHERE a.original_domain IN ('topology', 'linear_algebra', 'optimization',
                                     'optimal_transport', 'category_theory', 'differential_geometry')
          AND p.passage_text ILIKE '%' || a.name || '%'
        LIMIT 1000
    """)

    passages = cur.fetchall()

    flagged = defaultdict(list)
    for algo_id, name, text, passage_id in passages:
        issues = []
        for pattern in OCR_SUSPECT_PATTERNS:
            if re.search(pattern, text):
                issues.append(pattern)

        if issues:
            flagged[algo_id].append({
                'passage_id': str(passage_id),
                'issues': issues
            })

    # Update algorithms with OCR flags
    for algo_id, issues in flagged.items():
        cur.execute("""
            UPDATE algorithms
            SET ocr_quality_flag = 'suspect',
                ocr_quality_notes = %s
            WHERE algo_id = %s
        """, (json.dumps(issues[:5]), algo_id))  # Keep first 5 issues

    # Mark others as good
    cur.execute("""
        UPDATE algorithms
        SET ocr_quality_flag = 'good'
        WHERE ocr_quality_flag = 'unknown'
          AND algo_id NOT IN (SELECT algo_id FROM algorithms WHERE ocr_quality_flag = 'suspect')
    """)

    conn.commit()
    print(f"Flagged {len(flagged)} algorithms with potential OCR issues")
    return len(flagged)


def generate_spatial_applications(conn):
    """Generate spatial biology application suggestions for algorithms."""
    print("Generating spatial biology applications...")

    # Predefined mappings for key algorithm types
    spatial_mappings = {
        'topology': ['tissue architecture analysis', 'tumor boundary detection', 'cell neighborhood topology'],
        'optimal_transport': ['spatial alignment', 'cell fate trajectory', 'domain adaptation between samples'],
        'graph_theory': ['cell-cell interaction networks', 'spatial community detection', 'neighborhood analysis'],
        'clustering': ['cell type identification', 'spatial domain segmentation', 'spot clustering'],
        'decomposition': ['gene expression deconvolution', 'spatial factor analysis', 'cell type unmixing'],
        'signal_processing': ['image enhancement', 'noise reduction', 'feature extraction from H&E'],
        'compressed_sensing': ['sparse gene imputation', 'missing spot reconstruction', 'low-rank completion'],
        'control_theory': ['cell fate control', 'gene regulatory dynamics', 'perturbation response'],
        'game_theory': ['cell competition modeling', 'tumor evolution', 'immune-tumor dynamics'],
        'information_theory': ['gene selection', 'spatial information quantification', 'marker gene identification'],
    }

    cur = conn.cursor()

    updated = 0
    for domain, applications in spatial_mappings.items():
        cur.execute("""
            UPDATE algorithms
            SET spatial_biology_uses = %s
            WHERE original_domain = %s
              AND spatial_biology_uses IS NULL
        """, (applications, domain))
        updated += cur.rowcount

    conn.commit()
    print(f"Updated {updated} algorithms with spatial biology applications")
    return updated


def show_stats(conn):
    """Show algorithm registry statistics."""
    cur = conn.cursor()

    print("\n" + "="*60)
    print("ALGORITHM REGISTRY STATISTICS")
    print("="*60)

    # Total counts
    cur.execute("SELECT COUNT(*) FROM algorithms")
    total = cur.fetchone()[0]
    print(f"\nTotal algorithms: {total}")

    # By domain
    cur.execute("""
        SELECT original_domain, COUNT(*) as count
        FROM algorithms
        GROUP BY original_domain
        ORDER BY count DESC
        LIMIT 15
    """)
    print("\nBy domain:")
    for domain, count in cur.fetchall():
        print(f"  {domain or 'unclassified'}: {count}")

    # By category
    cur.execute("""
        SELECT category, COUNT(*) as count
        FROM algorithms
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10
    """)
    print("\nBy category:")
    for cat, count in cur.fetchall():
        print(f"  {cat or 'general'}: {count}")

    # OCR quality
    cur.execute("""
        SELECT ocr_quality_flag, COUNT(*) as count
        FROM algorithms
        GROUP BY ocr_quality_flag
    """)
    print("\nOCR quality:")
    for flag, count in cur.fetchall():
        print(f"  {flag}: {count}")

    # Paper links
    cur.execute("SELECT COUNT(*) FROM algorithm_papers")
    paper_links = cur.fetchone()[0]
    print(f"\nPaper links: {paper_links}")

    # Repo links
    cur.execute("SELECT COUNT(*) FROM algorithm_repos")
    repo_links = cur.fetchone()[0]
    print(f"Repository links: {repo_links}")

    # Polymathic bridges
    cur.execute("SELECT COUNT(*) FROM algorithm_bridges")
    bridges = cur.fetchone()[0]
    print(f"Polymathic bridges: {bridges}")

    # Top polymathic algorithms
    cur.execute("""
        SELECT a.name, a.original_domain, COUNT(ab.bridge_id) as bridges
        FROM algorithms a
        JOIN algorithm_bridges ab ON a.algo_id = ab.algo_id
        GROUP BY a.algo_id, a.name, a.original_domain
        ORDER BY bridges DESC
        LIMIT 10
    """)
    print("\nTop polymathic algorithms:")
    for name, domain, count in cur.fetchall():
        print(f"  {name} ({domain}): {count} bridges")

    # Algorithms needing OCR review
    cur.execute("""
        SELECT name, original_domain
        FROM algorithms
        WHERE ocr_quality_flag = 'suspect'
        ORDER BY mention_count DESC
        LIMIT 10
    """)
    print("\nAlgorithms needing OCR review (math-heavy):")
    for name, domain in cur.fetchall():
        print(f"  ⚠️  {name} ({domain})")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Build Algorithm Registry")
    parser.add_argument('--extract', action='store_true', help='Extract algorithms from concepts')
    parser.add_argument('--classify', action='store_true', help='Classify algorithms (placeholder for LLM)')
    parser.add_argument('--link', action='store_true', help='Link to papers and repos')
    parser.add_argument('--bridges', action='store_true', help='Find polymathic bridges')
    parser.add_argument('--ocr-audit', action='store_true', help='Audit OCR quality')
    parser.add_argument('--spatial', action='store_true', help='Generate spatial applications')
    parser.add_argument('--stats', action='store_true', help='Show registry statistics')
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--min-mentions', type=int, default=5, help='Minimum mentions for extraction')

    args = parser.parse_args()

    if not any([args.extract, args.classify, args.link, args.bridges,
                args.ocr_audit, args.spatial, args.stats, args.full]):
        parser.print_help()
        return

    conn = get_db_connection()

    try:
        if args.full or args.extract:
            extract_algorithms(conn, args.min_mentions)

        if args.full or args.link:
            link_to_repos(conn)

        if args.full or args.bridges:
            find_polymathic_bridges(conn)

        if args.full or args.ocr_audit:
            audit_ocr_quality(conn)

        if args.full or args.spatial:
            generate_spatial_applications(conn)

        if args.full or args.stats:
            show_stats(conn)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
