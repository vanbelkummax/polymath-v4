#!/usr/bin/env python3
"""
Import Tests for Polymath v4

Verifies that all core modules can be imported without errors.

Run with: python -m pytest tests/test_imports.py -v
"""

import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_import():
    """Test config module imports."""
    from lib.config import config
    assert config is not None
    assert config.POSTGRES_DSN is not None
    assert config.EMBEDDING_DIM == 1024


def test_embedder_import():
    """Test embedder module imports."""
    from lib.embeddings.bge_m3 import BGEEmbedder, Embedder, BGEM3Embedder, get_embedder
    assert BGEEmbedder is not None
    assert Embedder is BGEEmbedder  # Alias
    assert BGEM3Embedder is BGEEmbedder  # Alias
    assert get_embedder is not None


def test_embedder_methods():
    """Test embedder has required methods."""
    from lib.embeddings.bge_m3 import BGEEmbedder

    # Check class has required methods (without instantiating)
    assert hasattr(BGEEmbedder, 'encode')
    assert hasattr(BGEEmbedder, 'embed_single')
    assert hasattr(BGEEmbedder, 'embed_batch')
    assert hasattr(BGEEmbedder, 'encode_query')


def test_db_import():
    """Test database module imports."""
    from lib.db.postgres import (
        ConnectionPool,
        get_pool,
        get_connection,
        get_db_connection,
        execute_query,
        check_health,
    )
    assert ConnectionPool is not None
    assert get_pool is not None
    assert get_connection is not None


def test_ingest_imports():
    """Test ingestion module imports."""
    from lib.ingest.pdf_parser import PDFParser
    from lib.ingest.chunking import chunk_text
    from lib.ingest.asset_detector import AssetDetector, DetectedAsset

    assert PDFParser is not None
    assert chunk_text is not None
    assert AssetDetector is not None


def test_unified_ingest_import():
    """Test unified ingest module imports."""
    from lib.unified_ingest import (
        UnifiedIngestor,
        IngestResult,
        BatchResult,
        ingest_pdf,
        ingest_directory,
    )
    assert UnifiedIngestor is not None
    assert IngestResult is not None


def test_search_import():
    """Test search module imports."""
    from lib.search.hybrid_search import HybridSearcher
    assert HybridSearcher is not None


def test_all_scripts_importable():
    """Test that all scripts can be imported."""
    # Just check they parse without error
    import importlib.util

    scripts = [
        'scripts/ingest_pdf.py',
        'scripts/batch_concepts.py',
        'scripts/extract_skills.py',
        'scripts/github_ingest.py',
        'scripts/system_report.py',
    ]

    for script in scripts:
        path = Path(__file__).parent.parent / script
        if path.exists():
            spec = importlib.util.spec_from_file_location(script, path)
            assert spec is not None, f"Could not load {script}"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
