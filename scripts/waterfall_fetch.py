#!/usr/bin/env python3
"""
Waterfall Paper Fetcher

Attempts to retrieve papers by DOI using multiple sources in priority order:
1. CORE API (full text directly)
2. Unpaywall (OA PDF links)
3. OpenAlex (metadata + OA locations)
4. CrossRef (metadata only)
5. Web search (last resort, just logs the URL)

Usage:
    python scripts/waterfall_fetch.py --from-db --limit 20
    python scripts/waterfall_fetch.py --doi "10.1038/s41587-025-02895-3"
    python scripts/waterfall_fetch.py --file /tmp/missing_dois.txt --limit 10
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg2
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Load API keys from .env
def load_env():
    env_vars = {}
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    env_vars[key.strip()] = val.strip().strip('"').strip("'")
    return env_vars

ENV = load_env()

CORE_API_KEY = os.environ.get("CORE_API_KEY", ENV.get("CORE_API_KEY", ""))
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", ENV.get("UNPAYWALL_EMAIL", "max.vanbelkum@vanderbilt.edu"))
OPENALEX_EMAIL = os.environ.get("OPENALEX_EMAIL", ENV.get("OPENALEX_EMAIL", "max.vanbelkum@vanderbilt.edu"))

STAGING_DIR = Path("/home/user/work/polymax/ingest_staging")
STAGING_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Source Clients
# ============================================================================

class WaterfallFetcher:
    """Fetches papers via multiple sources in priority order."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Polymath/4.0 (mailto:max.vanbelkum@vanderbilt.edu)"
        })
        self.stats = {
            "core": 0,
            "unpaywall": 0,
            "openalex": 0,
            "crossref": 0,
            "web_search": 0,
            "failed": 0
        }

    def fetch_doi(self, doi: str) -> Dict:
        """
        Try to fetch paper by DOI using waterfall approach.
        Returns dict with: source, title, pdf_url, full_text, metadata
        """
        doi = doi.strip()
        logger.info(f"Fetching: {doi}")

        # 1. Try CORE API
        result = self._try_core(doi)
        if result:
            self.stats["core"] += 1
            return result

        # 2. Try Unpaywall
        result = self._try_unpaywall(doi)
        if result:
            self.stats["unpaywall"] += 1
            return result

        # 3. Try OpenAlex
        result = self._try_openalex(doi)
        if result:
            self.stats["openalex"] += 1
            return result

        # 4. Try CrossRef (metadata only)
        result = self._try_crossref(doi)
        if result:
            self.stats["crossref"] += 1
            return result

        # 5. Web search fallback
        result = self._web_search_fallback(doi)
        if result:
            self.stats["web_search"] += 1
            return result

        self.stats["failed"] += 1
        return None

    def _try_core(self, doi: str) -> Optional[Dict]:
        """Try CORE API for full text."""
        if not CORE_API_KEY:
            return None

        try:
            url = f"https://api.core.ac.uk/v3/search/works"
            params = {"q": f'doi:"{doi}"', "limit": 1}
            headers = {"Authorization": f"Bearer {CORE_API_KEY}"}

            resp = self.session.get(url, params=params, headers=headers, timeout=15)
            time.sleep(6.5)  # Rate limit

            if resp.status_code != 200:
                return None

            data = resp.json()
            results = data.get("results", [])

            if not results:
                return None

            work = results[0]
            full_text = work.get("fullText", "")

            if len(full_text) < 1000:
                return None

            logger.info(f"  ✓ CORE: Found full text ({len(full_text)} chars)")
            return {
                "source": "core",
                "doi": doi,
                "title": work.get("title", ""),
                "authors": [a.get("name", "") for a in work.get("authors", [])],
                "year": work.get("yearPublished"),
                "full_text": full_text,
                "pdf_url": work.get("downloadUrl")
            }
        except Exception as e:
            logger.debug(f"  CORE failed: {e}")
            return None

    def _try_unpaywall(self, doi: str) -> Optional[Dict]:
        """Try Unpaywall for OA PDF link."""
        try:
            url = f"https://api.unpaywall.org/v2/{doi}"
            params = {"email": UNPAYWALL_EMAIL}

            resp = self.session.get(url, params=params, timeout=15)
            time.sleep(1)

            if resp.status_code != 200:
                return None

            data = resp.json()

            # Find best OA location
            best_url = None
            for loc in data.get("oa_locations", []):
                if loc.get("url_for_pdf"):
                    best_url = loc["url_for_pdf"]
                    break
                elif loc.get("url"):
                    best_url = loc["url"]

            if not best_url:
                return None

            logger.info(f"  ✓ Unpaywall: Found OA link")
            return {
                "source": "unpaywall",
                "doi": doi,
                "title": data.get("title", ""),
                "authors": [a.get("family", "") + ", " + a.get("given", "")
                           for a in data.get("z_authors", []) or []],
                "year": data.get("year"),
                "pdf_url": best_url,
                "is_oa": data.get("is_oa", False)
            }
        except Exception as e:
            logger.debug(f"  Unpaywall failed: {e}")
            return None

    def _try_openalex(self, doi: str) -> Optional[Dict]:
        """Try OpenAlex for metadata and OA links."""
        try:
            url = f"https://api.openalex.org/works/doi:{doi}"
            params = {"mailto": OPENALEX_EMAIL}

            resp = self.session.get(url, params=params, timeout=15)
            time.sleep(0.5)

            if resp.status_code != 200:
                return None

            data = resp.json()

            # Get PDF URL from best OA location
            pdf_url = None
            for loc in data.get("open_access", {}).get("oa_locations", []) or []:
                if loc.get("pdf_url"):
                    pdf_url = loc["pdf_url"]
                    break

            if not pdf_url:
                pdf_url = data.get("open_access", {}).get("oa_url")

            if not pdf_url:
                # No OA, but we have metadata
                logger.info(f"  ~ OpenAlex: Metadata only (no OA)")
                return {
                    "source": "openalex_metadata",
                    "doi": doi,
                    "title": data.get("title", ""),
                    "authors": [a.get("author", {}).get("display_name", "")
                               for a in data.get("authorships", [])],
                    "year": data.get("publication_year"),
                    "pdf_url": None,
                    "cited_by": data.get("cited_by_count", 0)
                }

            logger.info(f"  ✓ OpenAlex: Found OA link")
            return {
                "source": "openalex",
                "doi": doi,
                "title": data.get("title", ""),
                "authors": [a.get("author", {}).get("display_name", "")
                           for a in data.get("authorships", [])],
                "year": data.get("publication_year"),
                "pdf_url": pdf_url,
                "cited_by": data.get("cited_by_count", 0)
            }
        except Exception as e:
            logger.debug(f"  OpenAlex failed: {e}")
            return None

    def _try_crossref(self, doi: str) -> Optional[Dict]:
        """Try CrossRef for metadata."""
        try:
            url = f"https://api.crossref.org/works/{doi}"
            headers = {"User-Agent": f"Polymath/4.0 (mailto:{UNPAYWALL_EMAIL})"}

            resp = self.session.get(url, headers=headers, timeout=15)
            time.sleep(1)

            if resp.status_code != 200:
                return None

            data = resp.json().get("message", {})

            title = data.get("title", [""])[0] if data.get("title") else ""

            logger.info(f"  ~ CrossRef: Metadata only")
            return {
                "source": "crossref",
                "doi": doi,
                "title": title,
                "authors": [f"{a.get('family', '')}, {a.get('given', '')}"
                           for a in data.get("author", [])],
                "year": data.get("published", {}).get("date-parts", [[None]])[0][0],
                "pdf_url": None,
                "venue": data.get("container-title", [""])[0] if data.get("container-title") else ""
            }
        except Exception as e:
            logger.debug(f"  CrossRef failed: {e}")
            return None

    def _web_search_fallback(self, doi: str) -> Optional[Dict]:
        """Generate web search URL as last resort."""
        search_url = f"https://scholar.google.com/scholar?q={doi}"
        logger.info(f"  → Web search: {search_url}")
        return {
            "source": "web_search",
            "doi": doi,
            "search_url": search_url,
            "pdf_url": None
        }

    def download_pdf(self, url: str, doi: str) -> Optional[Path]:
        """Download PDF to staging directory."""
        try:
            safe_name = re.sub(r'[^\w\-.]', '_', doi) + ".pdf"
            dest = STAGING_DIR / safe_name

            if dest.exists():
                logger.info(f"  PDF already exists: {dest.name}")
                return dest

            resp = self.session.get(url, timeout=60, stream=True)
            if resp.status_code != 200:
                return None

            # Check if it's actually a PDF
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type.lower() and not resp.content[:4] == b'%PDF':
                logger.warning(f"  Not a PDF: {content_type}")
                return None

            with open(dest, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"  ✓ Downloaded: {dest.name}")
            return dest
        except Exception as e:
            logger.warning(f"  Download failed: {e}")
            return None


# ============================================================================
# Database Integration
# ============================================================================

def get_missing_dois_from_db(limit: int = 50) -> List[Tuple[str, int]]:
    """Get most-cited missing DOIs from database."""
    conn = psycopg2.connect(config.POSTGRES_DSN)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                WITH extracted_dois AS (
                    SELECT
                        (regexp_matches(passage_text, '(10\\.1038/[a-zA-Z0-9\\.\\-]+[a-zA-Z0-9])', 'g'))[1] as doi
                    FROM passages
                    UNION ALL
                    SELECT
                        (regexp_matches(passage_text, '(10\\.1016/j\\.cell\\.\\d{4}\\.\\d+\\.\\d+)', 'g'))[1] as doi
                    FROM passages
                    UNION ALL
                    SELECT
                        (regexp_matches(passage_text, '(10\\.1126/science\\.[a-zA-Z0-9\\.]+)', 'g'))[1] as doi
                    FROM passages
                )
                SELECT e.doi, COUNT(*) as mentions
                FROM extracted_dois e
                WHERE e.doi IS NOT NULL
                  AND e.doi NOT IN (SELECT doi FROM documents WHERE doi IS NOT NULL)
                  AND e.doi NOT LIKE '%%-'
                  AND length(e.doi) > 15
                GROUP BY e.doi
                ORDER BY mentions DESC
                LIMIT %s
            """, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Waterfall paper fetcher")
    parser.add_argument("--doi", help="Single DOI to fetch")
    parser.add_argument("--file", help="File with DOIs (one per line)")
    parser.add_argument("--from-db", action="store_true", help="Get missing DOIs from database")
    parser.add_argument("--limit", type=int, default=20, help="Max DOIs to process")
    parser.add_argument("--download", action="store_true", help="Download PDFs to staging")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be fetched")
    args = parser.parse_args()

    fetcher = WaterfallFetcher()
    dois = []

    if args.doi:
        dois = [(args.doi, 0)]
    elif args.file:
        with open(args.file) as f:
            for line in f:
                parts = line.strip().split("|")
                if parts:
                    doi = parts[0].strip()
                    mentions = int(parts[1].strip()) if len(parts) > 1 else 0
                    if doi:
                        dois.append((doi, mentions))
        dois = dois[:args.limit]
    elif args.from_db:
        dois = get_missing_dois_from_db(args.limit)
    else:
        parser.print_help()
        return

    logger.info(f"Processing {len(dois)} DOIs...")

    if args.dry_run:
        for doi, mentions in dois:
            print(f"{doi} ({mentions} mentions)")
        return

    results = []
    for doi, mentions in dois:
        result = fetcher.fetch_doi(doi)
        if result:
            result["mentions"] = mentions
            results.append(result)

            # Download PDF if requested and available
            if args.download and result.get("pdf_url"):
                fetcher.download_pdf(result["pdf_url"], doi)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"CORE (full text):  {fetcher.stats['core']}")
    print(f"Unpaywall (OA):    {fetcher.stats['unpaywall']}")
    print(f"OpenAlex (OA):     {fetcher.stats['openalex']}")
    print(f"CrossRef (meta):   {fetcher.stats['crossref']}")
    print(f"Web search:        {fetcher.stats['web_search']}")
    print(f"Failed:            {fetcher.stats['failed']}")
    print("="*60)

    # Show results with PDFs
    pdf_available = [r for r in results if r.get("pdf_url")]
    if pdf_available:
        print(f"\nPapers with PDF available ({len(pdf_available)}):")
        for r in pdf_available[:10]:
            print(f"  [{r['source']}] {r.get('title', r['doi'])[:60]}...")


if __name__ == "__main__":
    main()
