#!/usr/bin/env python3
"""
Waterfall paper acquisition - tries multiple sources until success.

NOTE: Run with python -u for unbuffered output:
    python -u scripts/waterfall_acquire.py --polymathic --all

Sources (in order):
1. CORE API (open access aggregator)
2. Unpaywall (OA link finder)
3. arXiv direct
4. bioRxiv/medRxiv direct
5. Semantic Scholar (metadata + OA links)
6. → If all fail, add to manual list

Usage:
    python scripts/waterfall_acquire.py --query "topological data analysis biology" --limit 20
    python scripts/waterfall_acquire.py --doi "10.1038/s41592-024-02201-0"
    python scripts/waterfall_acquire.py --polymathic --field tda
    python scripts/waterfall_acquire.py --polymathic --all
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MANUAL_LIST_PATH = Path("/home/user/polymath-v4/data/manual_retrieval_needed.jsonl")
SUCCESS_LOG_PATH = Path("/home/user/polymath-v4/data/waterfall_success.jsonl")
STAGING_DIR = Path("/home/user/work/polymax/ingest_staging")

# Polymathic field queries
POLYMATHIC_QUERIES = {
    "tda": [
        "persistent homology single cell RNA-seq",
        "topological data analysis spatial transcriptomics",
        "mapper algorithm gene expression",
        "Betti numbers tissue architecture",
        "persistent homology cancer",
        "TDA machine learning biology",
    ],
    "sheaf": [
        "sheaf neural networks",
        "cellular sheaves graph learning",
        "category theory machine learning",
        "sheaf theory data fusion",
        "compositional deep learning",
    ],
    "game_theory": [
        "evolutionary game theory tumor",
        "game theory cell competition",
        "Nash equilibrium cancer evolution",
        "spatial games cell biology",
        "game theory immune response",
    ],
    "control": [
        "control theory gene regulatory networks",
        "optimal control cell fate",
        "feedback control synthetic biology",
        "control theory systems biology",
        "dynamical systems cell differentiation",
    ],
    "compressed_sensing": [
        "compressed sensing single cell",
        "sparse reconstruction RNA-seq",
        "compressive sensing genomics",
        "L1 minimization gene expression",
    ],
    "info_geometry": [
        "information geometry neural networks",
        "Fisher information deep learning",
        "natural gradient optimization",
        "statistical manifold learning",
    ],
    "tropical": [
        "tropical geometry optimization",
        "max-plus algebra biology",
        "tropical methods phylogenetics",
    ],
    "renormalization": [
        "renormalization group machine learning",
        "multiscale modeling cells",
        "coarse graining biological systems",
    ],
}


def log_success(paper_info: dict):
    """Log successful acquisition."""
    SUCCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUCCESS_LOG_PATH, "a") as f:
        paper_info["acquired_at"] = datetime.now().isoformat()
        f.write(json.dumps(paper_info) + "\n")


def log_manual_needed(paper_info: dict, reason: str):
    """Log paper that needs manual retrieval."""
    MANUAL_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_LIST_PATH, "a") as f:
        paper_info["reason"] = reason
        paper_info["logged_at"] = datetime.now().isoformat()
        f.write(json.dumps(paper_info) + "\n")
    print(f"  → Added to manual list: {paper_info.get('title', paper_info.get('doi', 'Unknown'))[:60]}")


def try_core_api(query: str, limit: int = 10) -> list:
    """Try CORE API for open access papers."""
    print(f"\n[1/5] Trying CORE API: {query[:50]}...")

    try:
        result = subprocess.run(
            [sys.executable, "scripts/discover_papers.py", query, "--limit", str(limit), "--json"],
            capture_output=True, text=True, timeout=60, cwd="/home/user/polymath-v4"
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse JSON output
            papers = []
            for line in result.stdout.strip().split("\n"):
                if line.startswith("{"):
                    try:
                        papers.append(json.loads(line))
                    except:
                        pass
            print(f"  Found {len(papers)} papers via CORE")
            return papers
    except Exception as e:
        print(f"  CORE API error: {e}")

    return []


def try_unpaywall(doi: str) -> str | None:
    """Try Unpaywall for OA PDF link."""
    print(f"  [2/5] Trying Unpaywall for {doi}...")

    email = "polymath@vanderbilt.edu"
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("is_oa") and data.get("best_oa_location"):
                pdf_url = data["best_oa_location"].get("url_for_pdf")
                if pdf_url:
                    print(f"  Found OA PDF via Unpaywall")
                    return pdf_url
    except Exception as e:
        print(f"  Unpaywall error: {e}")

    return None


def try_arxiv(query: str = None, arxiv_id: str = None) -> list:
    """Try arXiv API."""
    print(f"  [3/5] Trying arXiv...")

    if arxiv_id:
        # Direct arXiv ID lookup
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    elif query:
        url = f"http://export.arxiv.org/api/query?search_query=all:{quote(query)}&max_results=10"
    else:
        return []

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            # Parse Atom feed (simple extraction)
            entries = re.findall(r'<entry>(.*?)</entry>', resp.text, re.DOTALL)
            papers = []
            for entry in entries:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                arxiv_id = re.search(r'<id>http://arxiv.org/abs/(.*?)</id>', entry)
                if title and arxiv_id:
                    papers.append({
                        "title": title.group(1).strip().replace("\n", " "),
                        "arxiv_id": arxiv_id.group(1),
                        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id.group(1)}.pdf",
                        "source": "arxiv"
                    })
            if papers:
                print(f"  Found {len(papers)} papers on arXiv")
            return papers
    except Exception as e:
        print(f"  arXiv error: {e}")

    return []


def try_biorxiv(query: str) -> list:
    """Try bioRxiv/medRxiv API."""
    print(f"  [4/5] Trying bioRxiv/medRxiv...")

    # bioRxiv content API
    url = f"https://api.biorxiv.org/details/biorxiv/2020-01-01/2026-12-31/100"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            papers = []
            query_lower = query.lower()
            for paper in data.get("collection", []):
                if query_lower in paper.get("title", "").lower() or \
                   query_lower in paper.get("abstract", "").lower():
                    papers.append({
                        "title": paper.get("title"),
                        "doi": paper.get("doi"),
                        "pdf_url": f"https://www.biorxiv.org/content/{paper.get('doi')}.full.pdf",
                        "source": "biorxiv"
                    })
            if papers:
                print(f"  Found {len(papers)} papers on bioRxiv")
            return papers[:10]
    except Exception as e:
        print(f"  bioRxiv error: {e}")

    return []


def try_semantic_scholar(query: str = None, doi: str = None) -> list:
    """Try Semantic Scholar API."""
    print(f"  [5/5] Trying Semantic Scholar...")

    headers = {}
    s2_key = os.environ.get("S2_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key

    try:
        if doi:
            url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=title,authors,year,openAccessPdf"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("openAccessPdf"):
                    return [{
                        "title": data.get("title"),
                        "doi": doi,
                        "pdf_url": data["openAccessPdf"].get("url"),
                        "source": "semantic_scholar"
                    }]
        elif query:
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(query)}&limit=10&fields=title,authors,year,openAccessPdf,externalIds"
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                papers = []
                for paper in data.get("data", []):
                    if paper.get("openAccessPdf"):
                        papers.append({
                            "title": paper.get("title"),
                            "doi": paper.get("externalIds", {}).get("DOI"),
                            "pdf_url": paper["openAccessPdf"].get("url"),
                            "source": "semantic_scholar"
                        })
                if papers:
                    print(f"  Found {len(papers)} OA papers on S2")
                return papers
    except Exception as e:
        print(f"  Semantic Scholar error: {e}")

    return []


def download_pdf(url: str, filename: str) -> bool:
    """Download PDF to staging directory."""
    try:
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        filepath = STAGING_DIR / filename

        resp = requests.get(url, timeout=30, stream=True)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"  ✓ Downloaded: {filename}")
            return True
    except Exception as e:
        print(f"  Download failed: {e}")
    return False


def ingest_pdf(filepath: Path) -> bool:
    """Ingest PDF into Polymath."""
    try:
        result = subprocess.run(
            [sys.executable, "scripts/ingest_pdf.py", str(filepath)],
            capture_output=True, text=True, timeout=120, cwd="/home/user/polymath-v4"
        )
        if result.returncode == 0:
            print(f"  ✓ Ingested successfully")
            return True
        else:
            print(f"  Ingest warning: {result.stderr[:200] if result.stderr else 'Unknown'}")
            return "already exists" in result.stderr.lower() or "duplicate" in result.stderr.lower()
    except Exception as e:
        print(f"  Ingest error: {e}")
    return False


def discover_and_index_repos(doc_id: int = None):
    """Discover GitHub repos from paper text and index them."""
    try:
        # Run asset detector to find GitHub URLs in papers
        result = subprocess.run(
            [sys.executable, "scripts/detect_software_datasets.py", "--scan", "--limit", "100"],
            capture_output=True, text=True, timeout=300, cwd="/home/user/polymath-v4"
        )
        print(f"  Scanned for repos in new papers")

        # Index any new repos found
        result = subprocess.run(
            [sys.executable, "scripts/ingest_repos.py", "--source", "paper", "--limit", "50"],
            capture_output=True, text=True, timeout=600, cwd="/home/user/polymath-v4"
        )
        if "Saved:" in result.stdout:
            new_repos = result.stdout.count("Saved:")
            print(f"  ✓ Indexed {new_repos} new repos")
    except Exception as e:
        print(f"  Repo indexing note: {e}")


def waterfall_acquire(query: str = None, doi: str = None, limit: int = 10) -> dict:
    """
    Try multiple sources in waterfall fashion.
    Returns dict with success/failure counts.
    """
    results = {"success": 0, "manual_needed": 0, "already_have": 0}
    papers_to_try = []

    # Collect papers from all sources
    if query:
        # Try all sources for the query
        papers_to_try.extend(try_core_api(query, limit))
        papers_to_try.extend(try_arxiv(query=query))
        papers_to_try.extend(try_semantic_scholar(query=query))
        # bioRxiv is slow, skip for general queries

    if doi:
        # Try to get specific DOI
        pdf_url = try_unpaywall(doi)
        if pdf_url:
            papers_to_try.append({"doi": doi, "pdf_url": pdf_url, "source": "unpaywall"})

        s2_results = try_semantic_scholar(doi=doi)
        papers_to_try.extend(s2_results)

        # Check if it's an arXiv paper
        if "arxiv" in doi.lower():
            arxiv_id = doi.split("arxiv.")[-1]
            papers_to_try.extend(try_arxiv(arxiv_id=arxiv_id))

    # Deduplicate by title
    seen_titles = set()
    unique_papers = []
    for p in papers_to_try:
        title = p.get("title", "").lower()[:50]
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_papers.append(p)

    print(f"\n{'='*60}")
    print(f"Found {len(unique_papers)} unique papers to process")
    print(f"{'='*60}")

    # Try to download and ingest each
    for i, paper in enumerate(unique_papers[:limit], 1):
        title = paper.get("title", "Unknown")[:60]
        print(f"\n[{i}/{min(len(unique_papers), limit)}] {title}...")

        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            log_manual_needed(paper, "No PDF URL found")
            results["manual_needed"] += 1
            continue

        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_')
        filename = f"{safe_title}_{int(time.time())}.pdf"

        if download_pdf(pdf_url, filename):
            filepath = STAGING_DIR / filename
            if ingest_pdf(filepath):
                log_success(paper)
                results["success"] += 1
                # Clean up PDF after successful ingest
                try:
                    filepath.unlink()
                except:
                    pass
            else:
                # Check if it's a duplicate (still counts as "have")
                results["already_have"] += 1
        else:
            log_manual_needed(paper, f"Download failed from {paper.get('source', 'unknown')}")
            results["manual_needed"] += 1

        # Rate limiting
        time.sleep(0.5)

    return results


def run_polymathic_harvest(field: str = None, all_fields: bool = False):
    """Run polymathic harvest with waterfall acquisition."""

    fields_to_run = []
    if all_fields:
        fields_to_run = list(POLYMATHIC_QUERIES.keys())
    elif field:
        if field not in POLYMATHIC_QUERIES:
            print(f"Unknown field: {field}")
            print(f"Available: {', '.join(POLYMATHIC_QUERIES.keys())}")
            return
        fields_to_run = [field]

    total_results = {"success": 0, "manual_needed": 0, "already_have": 0}

    for f in fields_to_run:
        print(f"\n{'#'*60}")
        print(f"# HARVESTING: {f.upper()}")
        print(f"{'#'*60}")

        for query in POLYMATHIC_QUERIES[f]:
            print(f"\n>>> Query: {query}")
            results = waterfall_acquire(query=query, limit=10)
            for k, v in results.items():
                total_results[k] += v

            # Rate limit between queries
            time.sleep(1)

    print(f"\n{'='*60}")
    print(f"POLYMATHIC HARVEST COMPLETE")
    print(f"{'='*60}")
    print(f"  Successfully ingested: {total_results['success']}")
    print(f"  Already in KB:         {total_results['already_have']}")
    print(f"  Need manual retrieval: {total_results['manual_needed']}")

    if total_results['manual_needed'] > 0:
        print(f"\n  Manual list saved to: {MANUAL_LIST_PATH}")

    # Discover and index repos from newly ingested papers
    if total_results['success'] > 0:
        print(f"\n{'='*60}")
        print("Discovering and indexing repos from new papers...")
        print(f"{'='*60}")
        discover_and_index_repos()


def main():
    parser = argparse.ArgumentParser(description="Waterfall paper acquisition")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--doi", help="Specific DOI to fetch")
    parser.add_argument("--limit", type=int, default=10, help="Max papers per query")
    parser.add_argument("--polymathic", action="store_true", help="Run polymathic harvest")
    parser.add_argument("--field", help="Specific polymathic field")
    parser.add_argument("--all", action="store_true", help="All polymathic fields")
    parser.add_argument("--show-manual", action="store_true", help="Show manual retrieval list")

    args = parser.parse_args()

    if args.show_manual:
        if MANUAL_LIST_PATH.exists():
            print(f"Papers needing manual retrieval ({MANUAL_LIST_PATH}):\n")
            with open(MANUAL_LIST_PATH) as f:
                for line in f:
                    paper = json.loads(line)
                    print(f"  - {paper.get('title', 'Unknown')[:60]}")
                    print(f"    DOI: {paper.get('doi', 'N/A')}")
                    print(f"    Reason: {paper.get('reason', 'Unknown')}")
                    print()
        else:
            print("No manual retrieval list yet.")
        return

    if args.polymathic:
        run_polymathic_harvest(field=args.field, all_fields=args.all)
    elif args.query:
        waterfall_acquire(query=args.query, limit=args.limit)
    elif args.doi:
        waterfall_acquire(doi=args.doi)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
