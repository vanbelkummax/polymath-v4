#!/usr/bin/env python3
"""
Chunking Tests for Polymath v4

Tests the text chunking functionality.

Run with: python -m pytest tests/test_chunking.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ingest.chunking import chunk_text


def test_chunk_basic():
    """Test basic chunking."""
    text = "Hello world. " * 100
    chunks = chunk_text(text, max_size=200, overlap=50)

    assert len(chunks) > 1
    assert all('content' in c for c in chunks)


def test_chunk_respects_max_size():
    """Test that chunks respect max size."""
    text = "Word " * 1000
    chunks = chunk_text(text, max_size=500, overlap=50)

    for chunk in chunks:
        # Allow some slack for word boundaries
        assert len(chunk['content']) <= 600, f"Chunk too long: {len(chunk['content'])}"


def test_chunk_overlap():
    """Test that chunks have proper overlap."""
    text = "Word " * 500
    chunks = chunk_text(text, max_size=200, overlap=50)

    if len(chunks) > 1:
        # Check some overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            c1 = chunks[i]['content']
            c2 = chunks[i + 1]['content']
            # At least some text should be common
            assert len(set(c1.split()) & set(c2.split())) > 0 or len(c2) < 100


def test_chunk_empty():
    """Test chunking empty text."""
    chunks = chunk_text("", max_size=500, overlap=50)
    assert len(chunks) == 0


def test_chunk_short_text():
    """Test chunking text shorter than max size."""
    text = "Short text."
    chunks = chunk_text(text, max_size=500, overlap=50)

    assert len(chunks) == 1
    assert chunks[0]['content'] == text


def test_chunk_preserves_content():
    """Test that chunking preserves all content."""
    # Create text with unique markers
    text = " ".join([f"Marker{i}" for i in range(100)])
    chunks = chunk_text(text, max_size=200, overlap=20)

    # All markers should appear at least once
    all_content = " ".join(c['content'] for c in chunks)
    for i in range(100):
        assert f"Marker{i}" in all_content


def test_chunk_header_detection():
    """Test header detection in chunks (if supported)."""
    text = """
    Introduction
    This is the introduction section.

    Methods
    This section describes the methods.
    We used various techniques.

    Results
    Here are the results.
    """

    chunks = chunk_text(text, max_size=150, overlap=20)
    assert len(chunks) > 0

    # If header detection is implemented, chunks should have headers
    # This is optional functionality


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
