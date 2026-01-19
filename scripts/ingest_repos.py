#!/usr/bin/env python3
"""
Repository Ingestion Script

Indexes GitHub repositories with README + Python docstrings.
Links to papers when detected.

Usage:
    python scripts/ingest_repos.py                    # Ingest all sources
    python scripts/ingest_repos.py --source paper    # Only paper-linked
    python scripts/ingest_repos.py --source curated  # Only curated list
    python scripts/ingest_repos.py --limit 100       # Limit for testing
"""

import argparse
import ast
import base64
import hashlib
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from pathlib import Path

import requests

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.db.postgres import get_connection
from lib.embeddings.bge_m3 import BGEEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/home/user/logs/repo_ingest.log')
    ]
)
logger = logging.getLogger(__name__)

# GitHub API
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
GITHUB_API = 'https://api.github.com'
HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'Polymath-v4'
}
if GITHUB_TOKEN:
    HEADERS['Authorization'] = f'token {GITHUB_TOKEN}'

# False positive patterns to skip
FALSE_POSITIVE_PATTERNS = [
    r'^github\.com/(blog|posts|about|features|security|pricing|enterprise)$',
    r'^github\.com/[^/]+$',  # Single word without owner/repo
    r'^github\.com/orgs/',
    r'^github\.com/topics/',
]

# Curated spatial/ML repos
CURATED_REPOS = [
    # Spatial Transcriptomics
    'scverse/squidpy',
    'scverse/spatialdata',
    'scverse/scanpy',
    'theislab/scanpy',
    'satijalab/seurat',
    'mousepond/cellpose',
    'stardist/stardist',
    'BayraktarLab/cell2location',
    'almaan/stereoscope',

    # Cell-Cell Interaction
    'sqjin/CellChat',
    'Teichlab/cellphonedb',
    'saezlab/liana-py',
    'icbi-lab/liana',

    # Spatial Autocorrelation / Stats
    'xzhoulab/SPARK',
    'JEFworks-Lab/STdeconvolve',
    'Teichlab/SpatialDE',
    'MarioniLab/SpatialDE2',

    # Xenium / 10x
    '10XGenomics/xeniumranger',
    'pachterlab/monod',

    # Deconvolution
    'broadinstitute/Tangram',
    'MarcElosworthy/SpotClean',
    'drighelli/SpatialExperiment',

    # ST Prediction
    'mahmoodlab/HIPT',
    'mahmoodlab/CLAM',
    'owkin/HistoSSLscaling',
    'KatherLab/HIA',
    'biototem/TCGA-ST',

    # Single-cell
    'theislab/scvelo',
    'theislab/cellrank',
    'YosefLab/scvi-tools',
    'broadinstitute/infercnv',

    # Foundation Models
    'huggingface/transformers',
    'facebookresearch/dinov2',
    'openai/CLIP',
    'mlfoundations/open_clip',

    # ML Infrastructure
    'pytorch/pytorch',
    'numpy/numpy',
    'pandas-dev/pandas',
    'scikit-learn/scikit-learn',

    # Agentic Memory
    'mem0ai/mem0',
    'Dao-AILab/flash-attention',
    'letta-ai/letta',
    'memvid/memvid',
    'MemoriLabs/Memori',
    'steveyegge/beads',
]


def is_false_positive(url: str) -> bool:
    """Check if URL is a false positive."""
    # Normalize URL
    url = url.lower().replace('https://', '').replace('http://', '').rstrip('/')

    for pattern in FALSE_POSITIVE_PATTERNS:
        if re.match(pattern, url):
            return True

    # Check for valid owner/repo structure
    parts = url.replace('github.com/', '').split('/')
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return True

    return False


def parse_github_url(url: str) -> Optional[tuple]:
    """Extract owner and repo name from GitHub URL."""
    url = url.lower().replace('https://', '').replace('http://', '').rstrip('/')

    # Remove github.com prefix
    if url.startswith('github.com/'):
        url = url[11:]

    parts = url.split('/')
    if len(parts) >= 2:
        owner = parts[0]
        repo = parts[1].split('#')[0].split('?')[0]  # Remove anchors/params
        if owner and repo:
            return (owner, repo)

    return None


def fetch_github_metadata(owner: str, repo: str) -> Optional[dict]:
    """Fetch repo metadata from GitHub API."""
    url = f'{GITHUB_API}/repos/{owner}/{repo}'

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            logger.warning(f"Repo not found: {owner}/{repo}")
            return None
        if resp.status_code == 403:
            logger.warning(f"Rate limited, sleeping 60s")
            time.sleep(60)
            return fetch_github_metadata(owner, repo)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Error fetching {owner}/{repo}: {e}")
        return None


def fetch_readme(owner: str, repo: str) -> Optional[str]:
    """Fetch README content from GitHub API."""
    url = f'{GITHUB_API}/repos/{owner}/{repo}/readme'

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
        return content
    except Exception as e:
        logger.error(f"Error fetching README for {owner}/{repo}: {e}")
        return None


def fetch_python_files(owner: str, repo: str, branch: str = 'main') -> list:
    """Fetch list of Python files in repo."""
    url = f'{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1'

    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        if resp.status_code == 404:
            # Try master branch
            if branch == 'main':
                return fetch_python_files(owner, repo, 'master')
            return []
        resp.raise_for_status()
        data = resp.json()

        py_files = []
        for item in data.get('tree', []):
            if item['type'] == 'blob' and item['path'].endswith('.py'):
                # Skip test files and __pycache__
                if '/test' in item['path'] or '__pycache__' in item['path']:
                    continue
                # Prioritize src/, lib/, core files
                py_files.append(item['path'])

        # Limit to top 20 most important files
        priority_files = []
        other_files = []
        for f in py_files:
            if any(x in f for x in ['__init__', 'core', 'main', 'api', 'utils']):
                priority_files.append(f)
            else:
                other_files.append(f)

        return (priority_files + other_files)[:20]
    except Exception as e:
        logger.error(f"Error fetching file tree for {owner}/{repo}: {e}")
        return []


def fetch_file_content(owner: str, repo: str, path: str, branch: str = 'main') -> Optional[str]:
    """Fetch raw file content."""
    url = f'https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}'

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404 and branch == 'main':
            return fetch_file_content(owner, repo, path, 'master')
        resp.raise_for_status()
        return resp.text
    except:
        return None


def extract_docstrings(code: str, file_path: str) -> list:
    """Extract docstrings from Python code using AST."""
    docstrings = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Module docstring
    if ast.get_docstring(tree):
        docstrings.append({
            'text': ast.get_docstring(tree),
            'section': 'module_doc',
            'file_path': file_path,
            'function_name': None,
            'class_name': None
        })

    for node in ast.walk(tree):
        # Class docstrings
        if isinstance(node, ast.ClassDef):
            docstring = ast.get_docstring(node)
            if docstring and len(docstring) > 50:  # Skip trivial docstrings
                docstrings.append({
                    'text': docstring,
                    'section': 'class_doc',
                    'file_path': file_path,
                    'function_name': None,
                    'class_name': node.name
                })

        # Function docstrings
        elif isinstance(node, ast.FunctionDef):
            # Skip private/dunder methods
            if node.name.startswith('_') and not node.name.startswith('__init__'):
                continue

            docstring = ast.get_docstring(node)
            if docstring and len(docstring) > 30:
                # Get parent class if any
                class_name = None
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.walk(parent):
                            class_name = parent.name
                            break

                docstrings.append({
                    'text': docstring,
                    'section': 'docstring',
                    'file_path': file_path,
                    'function_name': node.name,
                    'class_name': class_name
                })

    return docstrings


def chunk_readme(readme: str, max_chars: int = 1000) -> list:
    """Chunk README into passages."""
    if not readme:
        return []

    chunks = []

    # Split by headers
    sections = re.split(r'\n##?\s+', readme)

    for section in sections:
        section = section.strip()
        if len(section) < 50:
            continue

        if len(section) <= max_chars:
            chunks.append({
                'text': section,
                'section': 'readme',
                'file_path': 'README.md',
                'function_name': None,
                'class_name': None
            })
        else:
            # Split long sections by paragraphs
            paragraphs = section.split('\n\n')
            current_chunk = ''
            for para in paragraphs:
                if len(current_chunk) + len(para) < max_chars:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk.strip():
                        chunks.append({
                            'text': current_chunk.strip(),
                            'section': 'readme',
                            'file_path': 'README.md',
                            'function_name': None,
                            'class_name': None
                        })
                    current_chunk = para + '\n\n'

            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'section': 'readme',
                    'file_path': 'README.md',
                    'function_name': None,
                    'class_name': None
                })

    return chunks


def collect_repo_urls(source: str = 'all') -> list:
    """Collect repo URLs from various sources."""
    urls = set()

    with get_connection() as conn:
        with conn.cursor() as cur:
            if source in ('all', 'paper'):
                # Paper-linked repos
                cur.execute("SELECT DISTINCT repo_url FROM paper_repos")
                for row in cur.fetchall():
                    urls.add(row[0])
                logger.info(f"Collected {len(urls)} paper-linked repos")

            if source in ('all', 'orphaned'):
                # Orphaned repos
                cur.execute("SELECT DISTINCT repo_url FROM archive.paper_repos_orphaned")
                for row in cur.fetchall():
                    urls.add(row[0])
                logger.info(f"Total after orphaned: {len(urls)} repos")

    if source in ('all', 'curated'):
        # Curated repos
        for repo in CURATED_REPOS:
            urls.add(f'https://github.com/{repo}')
        logger.info(f"Total after curated: {len(urls)} repos")

    # Filter false positives
    filtered = [u for u in urls if not is_false_positive(u)]
    logger.info(f"After filtering false positives: {len(filtered)} repos")

    return list(filtered)


def ingest_repo(url: str, embedder: BGEEmbedder) -> Optional[dict]:
    """Ingest a single repository."""
    parsed = parse_github_url(url)
    if not parsed:
        logger.warning(f"Could not parse URL: {url}")
        return None

    owner, repo = parsed

    # Fetch metadata
    metadata = fetch_github_metadata(owner, repo)
    if not metadata:
        return None

    # Fetch README
    readme = fetch_readme(owner, repo)

    # Fetch Python docstrings
    branch = metadata.get('default_branch', 'main')
    py_files = fetch_python_files(owner, repo, branch)

    docstrings = []
    for py_file in py_files[:15]:  # Limit files
        content = fetch_file_content(owner, repo, py_file, branch)
        if content:
            docstrings.extend(extract_docstrings(content, py_file))

    # Create passages
    passages = chunk_readme(readme) + docstrings

    if not passages:
        logger.warning(f"No passages extracted for {owner}/{repo}")
        # Still create the repo entry with metadata

    # Compute embeddings
    if passages:
        texts = [p['text'] for p in passages]
        embeddings = embedder.encode(texts)
        for i, passage in enumerate(passages):
            # Convert to list for pgvector
            passage['embedding'] = embeddings[i].tolist()

    return {
        'url': f'https://github.com/{owner}/{repo}',
        'owner': owner,
        'name': repo,
        'description': metadata.get('description'),
        'language': metadata.get('language'),
        'stars': metadata.get('stargazers_count', 0),
        'forks': metadata.get('forks_count', 0),
        'topics': metadata.get('topics', []),
        'default_branch': branch,
        'readme_content': readme,
        'github_id': metadata.get('id'),
        'passages': passages
    }


def save_repo(repo_data: dict, source_method: str) -> Optional[str]:
    """Save repository and passages to database."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Check if already exists
                cur.execute("SELECT repo_id FROM repositories WHERE repo_url = %s", (repo_data['url'],))
                existing = cur.fetchone()
                if existing:
                    logger.info(f"Repo already exists: {repo_data['url']}")
                    return existing[0]

                # Insert repository
                cur.execute("""
                    INSERT INTO repositories (
                        repo_url, owner, name, description, language,
                        stars, forks, topics, default_branch,
                        readme_content, github_id, source_method, last_github_sync
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING repo_id
                """, (
                    repo_data['url'],
                    repo_data['owner'],
                    repo_data['name'],
                    repo_data['description'],
                    repo_data['language'],
                    repo_data['stars'],
                    repo_data['forks'],
                    repo_data['topics'],
                    repo_data['default_branch'],
                    repo_data['readme_content'],
                    repo_data['github_id'],
                    source_method
                ))
                repo_id = cur.fetchone()[0]

                # Insert passages
                for passage in repo_data.get('passages', []):
                    cur.execute("""
                        INSERT INTO repo_passages (
                            repo_id, passage_text, section, file_path,
                            function_name, class_name, embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        repo_id,
                        passage['text'],
                        passage['section'],
                        passage.get('file_path'),
                        passage.get('function_name'),
                        passage.get('class_name'),
                        passage.get('embedding')
                    ))

                conn.commit()
                logger.info(f"Saved: {repo_data['owner']}/{repo_data['name']} ({len(repo_data.get('passages', []))} passages)")
                return repo_id

            except Exception as e:
                conn.rollback()
                logger.error(f"Error saving {repo_data['url']}: {e}")
                return None


def link_papers_to_repos():
    """Create paper-repo links from existing paper_repos table."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO paper_repo_links (doc_id, repo_id, link_type, confidence)
                    SELECT DISTINCT pr.doc_id, r.repo_id, 'mentioned', pr.confidence
                    FROM paper_repos pr
                    JOIN repositories r ON r.repo_url = pr.repo_url
                    ON CONFLICT (doc_id, repo_id) DO NOTHING
                """)
                count = cur.rowcount
                conn.commit()
                logger.info(f"Created {count} paper-repo links")
            except Exception as e:
                conn.rollback()
                logger.error(f"Error linking papers to repos: {e}")


def main():
    parser = argparse.ArgumentParser(description='Ingest GitHub repositories')
    parser.add_argument('--source', choices=['all', 'paper', 'orphaned', 'curated'],
                        default='all', help='Which repos to ingest')
    parser.add_argument('--limit', type=int, help='Limit number of repos')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    args = parser.parse_args()

    logger.info(f"Starting repo ingestion: source={args.source}, limit={args.limit}")

    # Collect URLs
    urls = collect_repo_urls(args.source)
    if args.limit:
        urls = urls[:args.limit]

    logger.info(f"Processing {len(urls)} repositories")

    # Initialize embedder
    embedder = BGEEmbedder()

    # Process repos
    success = 0
    failed = 0

    for i, url in enumerate(urls):
        try:
            logger.info(f"[{i+1}/{len(urls)}] Processing: {url}")

            # Determine source method
            if 'github.com/' in url:
                parsed = parse_github_url(url)
                if parsed and f"{parsed[0]}/{parsed[1]}" in [r.lower() for r in CURATED_REPOS]:
                    source_method = 'curated'
                else:
                    source_method = 'paper_detection'
            else:
                source_method = 'paper_detection'

            repo_data = ingest_repo(url, embedder)
            if repo_data:
                save_repo(repo_data, source_method)
                success += 1
            else:
                failed += 1

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            failed += 1

    # Link papers to repos
    logger.info("Linking papers to repositories...")
    link_papers_to_repos()

    logger.info(f"""
============================================================
REPO INGESTION COMPLETE
============================================================
Total: {len(urls)}
Succeeded: {success}
Failed: {failed}
============================================================
""")


if __name__ == '__main__':
    main()
