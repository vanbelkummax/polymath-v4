"""
Polymath v4 Configuration

Central configuration for all components.
"""

import os
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    SKILLS_DIR: Path = BASE_DIR / "skills"
    SKILLS_DRAFTS_DIR: Path = BASE_DIR / "skills_drafts"
    REPOS_DIR: Path = Path("/home/user/work/polymax/data/github_repos")

    # Database
    POSTGRES_DSN: str = os.environ.get(
        "POSTGRES_DSN",
        "dbname=polymath user=polymath host=/var/run/postgresql"
    )

    # Neo4j
    NEO4J_URI: str = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.environ.get("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "polymathic2026")

    # Google Cloud / Gemini
    GCP_PROJECT: str = os.environ.get("GCP_PROJECT", "fifth-branch-483806-m1")
    GCP_LOCATION: str = os.environ.get("GCP_LOCATION", "us-central1")
    GCS_BUCKET: str = os.environ.get("GCS_BUCKET", "polymath-batch-jobs")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/home/user/.gcp/service-account.json"
    )

    # Models
    GEMINI_MODEL: str = "gemini-2.5-flash-lite-preview-06-17"  # Batch API
    GEMINI_REALTIME_MODEL: str = "gemini-2.5-flash-preview-05-20"  # Real-time
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIM: int = 1024
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"

    # Processing
    NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "4"))
    BATCH_SIZE: int = 32
    MAX_PASSAGE_LENGTH: int = 2000
    CHUNK_OVERLAP: int = 200

    # GitHub
    GITHUB_TOKEN: str = os.environ.get("GITHUB_TOKEN", "")

    # HuggingFace
    HF_TOKEN: str = os.environ.get("HF_TOKEN", "")


config = Config()

# Ensure directories exist
config.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
config.SKILLS_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
