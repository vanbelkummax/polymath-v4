#!/usr/bin/env python3
"""Fast ENTITY sync with larger batches."""
import sys
sys.path.insert(0, '/home/user/polymath-v4')

import logging
from neo4j import GraphDatabase
from lib.db.postgres import get_pool
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 5000

def main():
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'polymathic2026'))
    pool = get_pool()

    with pool.connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('''
                SELECT DISTINCT pc.concept_name
                FROM passage_concepts pc
                JOIN passages p ON pc.passage_id = p.passage_id
                WHERE pc.concept_type = 'entity'
                AND (p.is_superseded = FALSE OR p.is_superseded IS NULL)
            ''')
            concepts = [row['concept_name'] for row in cur.fetchall()]

    total = len(concepts)
    logger.info(f'Syncing {total} ENTITY nodes with batch size {BATCH_SIZE}...')

    for i in range(0, total, BATCH_SIZE):
        batch = concepts[i:i + BATCH_SIZE]

        driver.execute_query(
            """
            UNWIND $names as name
            MERGE (c:ENTITY {name: name})
            SET c.synced_at = datetime()
            """,
            names=batch,
        )

        progress = min(i + len(batch), total)
        if progress % 25000 == 0 or progress == total:
            logger.info(f'  Progress: {progress:,}/{total:,} ({progress/total*100:.0f}%)')

    driver.close()
    logger.info('ENTITY sync complete!')

if __name__ == '__main__':
    main()
