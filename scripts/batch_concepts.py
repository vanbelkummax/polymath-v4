#!/usr/bin/env python3
"""
Batch Concept Extraction for Polymath v4

Uses Gemini Batch API (50% cost savings) to extract concepts from passages.

Usage:
    python scripts/batch_concepts.py --submit --limit 1000
    python scripts/batch_concepts.py --status
    python scripts/batch_concepts.py --process
"""

import argparse
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import psycopg2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Concept extraction prompt
CONCEPT_PROMPT = """Extract scientific concepts from this passage. Return JSON with these categories:

- methods: Techniques, algorithms, procedures (e.g., "attention mechanism", "gradient descent")
- problems: Research challenges being addressed (e.g., "cell type classification")
- domains: Research areas and fields (e.g., "spatial transcriptomics", "computer vision")
- datasets: Data sources mentioned (e.g., "ImageNet", "Visium HD")
- entities: Specific things like tools, diseases, genes (e.g., "PyTorch", "GBM", "TP53")

For each concept, include:
- name: The concept name (lowercase, normalized)
- confidence: 0.0-1.0 how certain you are

Return ONLY valid JSON, no markdown:
{{"methods": [{{"name": "...", "confidence": 0.9}}], "problems": [...], ...}}

PASSAGE:
{passage}
"""


def get_db_connection():
    return psycopg2.connect(config.POSTGRES_DSN)


def get_pending_passages(conn, limit: int = 1000) -> List[Dict]:
    """Get passages without concepts."""
    cur = conn.cursor()
    cur.execute("""
        SELECT p.passage_id, p.passage_text, d.title
        FROM passages p
        JOIN documents d ON p.doc_id = d.doc_id
        WHERE p.passage_id NOT IN (
            SELECT DISTINCT passage_id FROM passage_concepts
        )
        AND LENGTH(p.passage_text) > 100
        AND p.is_superseded = FALSE
        ORDER BY d.created_at DESC
        LIMIT %s
    """, (limit,))

    return [
        {'passage_id': str(row[0]), 'passage_text': row[1], 'title': row[2]}
        for row in cur.fetchall()
    ]


def create_batch_request(passages: List[Dict]) -> List[Dict]:
    """Create batch request format for Vertex AI."""
    requests = []
    for p in passages:
        prompt = CONCEPT_PROMPT.format(passage=p['passage_text'][:3000])
        requests.append({
            "request": {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json"
                }
            },
            "metadata": {
                "passage_id": p['passage_id'],
                "title": p['title'][:100]
            }
        })
    return requests


def submit_batch_job(conn, passages: List[Dict]) -> str:
    """Submit batch job using Google GenAI SDK."""
    from google.cloud import storage
    from google import genai
    from google.genai import types

    # Initialize GenAI client with Vertex AI
    client = genai.Client(
        vertexai=True,
        project=config.GCP_PROJECT,
        location=config.GCP_LOCATION
    )

    # Create request file
    job_id = f"concepts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create JSONL content for batch API
    jsonl_lines = []
    passage_mapping = {}  # Map request index to passage_id

    for i, p in enumerate(passages):
        prompt = CONCEPT_PROMPT.format(passage=p['passage_text'][:3000])
        # Vertex AI batch format requires "request" wrapper
        request = {
            "request": {
                "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048
                }
            }
        }
        jsonl_lines.append(json.dumps(request))
        passage_mapping[str(i)] = p['passage_id']

    # Upload to GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(config.GCS_BUCKET)

    input_path = f"batch_input/{job_id}.jsonl"
    input_uri = f"gs://{config.GCS_BUCKET}/{input_path}"

    blob = bucket.blob(input_path)
    blob.upload_from_string('\n'.join(jsonl_lines))
    logger.info(f"Uploaded {len(passages)} requests to {input_uri}")

    # Save passage mapping for result processing
    mapping_blob = bucket.blob(f"batch_input/{job_id}_mapping.json")
    mapping_blob.upload_from_string(json.dumps(passage_mapping, indent=2))

    # Submit batch job using GenAI SDK
    try:
        job = client.batches.create(
            model=config.GEMINI_MODEL,
            src=input_uri,
            config=types.CreateBatchJobConfig(
                display_name=f"polymath_concepts_{job_id}"
            )
        )
        job_name = job.name
        logger.info(f"Submitted batch job: {job_name}")
        logger.info(f"Initial state: {job.state}")
    except Exception as e:
        logger.error(f"Failed to submit batch job: {e}")
        raise

    # Track in database
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO concept_batch_jobs (job_id, gcs_input_uri, gcs_output_uri, passage_count, status)
        VALUES (%s, %s, %s, %s, 'running')
    """, (job_name, input_uri, f"gs://{config.GCS_BUCKET}/batch_output/{job_id}/", len(passages)))
    conn.commit()

    return job_name


def check_job_status(conn) -> List[Dict]:
    """Check status of pending jobs."""
    cur = conn.cursor()
    cur.execute("""
        SELECT job_id, gcs_output_uri, passage_count, status, submitted_at
        FROM concept_batch_jobs
        WHERE status IN ('pending', 'running')
        ORDER BY submitted_at DESC
    """)

    jobs = []
    for row in cur.fetchall():
        job_id, output_uri, count, status, submitted = row

        # Check GCP status
        try:
            from google.cloud import aiplatform
            aiplatform.init(project=config.GCP_PROJECT, location=config.GCP_LOCATION)
            job = aiplatform.BatchPredictionJob(job_id)
            gcp_status = job.state.name

            if gcp_status == 'JOB_STATE_SUCCEEDED':
                cur.execute("""
                    UPDATE concept_batch_jobs SET status = 'succeeded', completed_at = NOW()
                    WHERE job_id = %s
                """, (job_id,))
                conn.commit()
                status = 'succeeded'
            elif gcp_status == 'JOB_STATE_FAILED':
                cur.execute("""
                    UPDATE concept_batch_jobs SET status = 'failed', completed_at = NOW()
                    WHERE job_id = %s
                """, (job_id,))
                conn.commit()
                status = 'failed'

        except Exception as e:
            logger.warning(f"Could not check job {job_id}: {e}")

        jobs.append({
            'job_id': job_id,
            'output_uri': output_uri,
            'passages': count,
            'status': status,
            'submitted': submitted.isoformat() if submitted else None
        })

    return jobs


def process_results(conn, job_id: str = None) -> int:
    """Process completed batch job results."""
    from google.cloud import storage

    cur = conn.cursor()

    if job_id:
        cur.execute("""
            SELECT job_id, gcs_output_uri FROM concept_batch_jobs
            WHERE job_id = %s AND status = 'succeeded'
        """, (job_id,))
    else:
        # Get latest succeeded job
        cur.execute("""
            SELECT job_id, gcs_output_uri FROM concept_batch_jobs
            WHERE status = 'succeeded'
            ORDER BY completed_at DESC LIMIT 1
        """)

    row = cur.fetchone()
    if not row:
        logger.warning("No completed jobs to process")
        return 0

    job_id, output_uri = row
    logger.info(f"Processing results from {output_uri}")

    # Download results from GCS
    client = storage.Client()
    bucket_name = output_uri.split('/')[2]
    prefix = '/'.join(output_uri.split('/')[3:])

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    concepts_added = 0

    for blob in blobs:
        if not blob.name.endswith('.jsonl'):
            continue

        content = blob.download_as_text()
        for line in content.strip().split('\n'):
            if not line:
                continue

            try:
                result = json.loads(line)
                passage_id = result.get('metadata', {}).get('passage_id')
                response = result.get('response', {})

                # Extract concepts from response
                if 'candidates' in response:
                    text = response['candidates'][0]['content']['parts'][0]['text']
                    concepts = json.loads(text)

                    for concept_type in ['methods', 'problems', 'domains', 'datasets', 'entities']:
                        for c in concepts.get(concept_type, []):
                            name = c.get('name', c) if isinstance(c, dict) else c
                            confidence = c.get('confidence', 0.8) if isinstance(c, dict) else 0.8

                            cur.execute("""
                                INSERT INTO passage_concepts
                                (passage_id, concept_name, concept_type, confidence, extractor_model, extractor_version)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (passage_id, concept_name) DO NOTHING
                            """, (
                                passage_id,
                                name.lower().strip(),
                                concept_type.rstrip('s'),  # methods -> method
                                confidence,
                                config.GEMINI_MODEL,
                                'batch-v4'
                            ))
                            concepts_added += 1

            except Exception as e:
                logger.debug(f"Error processing result: {e}")

    conn.commit()

    # Mark job as processed
    cur.execute("""
        UPDATE concept_batch_jobs SET status = 'processed' WHERE job_id = %s
    """, (job_id,))
    conn.commit()

    logger.info(f"Added {concepts_added} concepts from job {job_id}")
    return concepts_added


def main():
    parser = argparse.ArgumentParser(description='Batch concept extraction')
    parser.add_argument('--submit', action='store_true', help='Submit new batch job')
    parser.add_argument('--status', action='store_true', help='Check job status')
    parser.add_argument('--process', action='store_true', help='Process completed results')
    parser.add_argument('--limit', type=int, default=1000, help='Max passages to process')
    parser.add_argument('--job-id', help='Specific job ID for --process')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    conn = get_db_connection()

    if args.submit:
        passages = get_pending_passages(conn, args.limit)
        if not passages:
            print("No passages pending concept extraction")
            return

        print(f"Found {len(passages)} passages to process")
        job_id = submit_batch_job(conn, passages)
        print(f"\nSubmitted job: {job_id}")
        print("Use --status to check progress")

    elif args.status:
        jobs = check_job_status(conn)
        if not jobs:
            print("No pending or running jobs")
            return

        print(f"\n{'Job ID':<50} {'Passages':<10} {'Status':<12} {'Submitted':<20}")
        print("-" * 92)
        for job in jobs:
            print(f"{job['job_id']:<50} {job['passages']:<10} {job['status']:<12} {job['submitted'] or '':<20}")

    elif args.process:
        count = process_results(conn, args.job_id)
        print(f"Added {count} concepts")

    else:
        # Show summary
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM passages WHERE passage_id NOT IN (SELECT DISTINCT passage_id FROM passage_concepts)")
        pending = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM passage_concepts")
        total = cur.fetchone()[0]

        print(f"\nConcept extraction status:")
        print(f"  Total concepts: {total:,}")
        print(f"  Passages pending: {pending:,}")
        print(f"\nUse --submit to start extraction")

    conn.close()


if __name__ == '__main__':
    main()
