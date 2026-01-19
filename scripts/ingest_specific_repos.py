#!/usr/bin/env python3
"""Ingest specific GitHub repos."""
import sys
sys.path.insert(0, '/home/user/polymath-v4')

import logging
from scripts.ingest_repos import ingest_repo, HEADERS
from lib.embeddings.bge_m3 import BGEEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

REPOS_TO_INGEST = [
    'https://github.com/mem0ai/mem0',
    'https://github.com/letta-ai/letta',
    'https://github.com/memvid/memvid',
    'https://github.com/MemoriLabs/Memori',
    'https://github.com/steveyegge/beads',
]

def main():
    embedder = BGEEmbedder()

    for url in REPOS_TO_INGEST:
        logger.info(f"Ingesting {url}...")
        try:
            result = ingest_repo(url, embedder)
            if result:
                logger.info(f"  ✓ {result.get('name', 'unknown')}: {result.get('passages', 0)} passages")
            else:
                logger.warning(f"  ✗ Failed to ingest {url}")
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")

    logger.info("Done!")

if __name__ == '__main__':
    main()
