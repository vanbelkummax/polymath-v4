#!/usr/bin/env python3
"""
Asset Detector Tests for Polymath v4

Tests the asset detection functionality.

Run with: python -m pytest tests/test_asset_detector.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.ingest.asset_detector import AssetDetector, DetectedAsset


def test_detect_github_url():
    """Test GitHub URL detection."""
    detector = AssetDetector()

    text = "We used code from https://github.com/pytorch/pytorch for training."
    assets = detector.detect_from_text(text)

    assert len(assets['github']) == 1
    assert 'pytorch/pytorch' in assets['github'][0].identifier


def test_detect_github_short_url():
    """Test GitHub URL without https."""
    detector = AssetDetector()

    text = "The code is available at github.com/scverse/squidpy"
    assets = detector.detect_from_text(text)

    assert len(assets['github']) == 1
    assert 'squidpy' in assets['github'][0].identifier


def test_detect_huggingface_url():
    """Test HuggingFace URL detection."""
    detector = AssetDetector()

    text = "We loaded the model from huggingface.co/facebook/dinov2-base"
    assets = detector.detect_from_text(text)

    assert len(assets['huggingface']) >= 1
    found_ids = [a.identifier for a in assets['huggingface']]
    assert any('dinov2' in id.lower() for id in found_ids)


def test_detect_from_pretrained():
    """Test detection of from_pretrained patterns."""
    detector = AssetDetector()

    text = 'model = AutoModel.from_pretrained("microsoft/deberta-v3-base")'
    assets = detector.detect_from_text(text)

    assert len(assets['huggingface']) >= 1


def test_detect_known_models():
    """Test detection of known model names via HuggingFace URLs."""
    detector = AssetDetector()

    # Test with explicit HuggingFace reference
    text = "We used bert-base-uncased from huggingface.co/bert-base-uncased"
    assets = detector.detect_from_text(text)

    # Should detect HuggingFace URL reference
    assert len(assets['huggingface']) >= 1


def test_detect_doi():
    """Test DOI detection."""
    detector = AssetDetector()

    text = "Our approach builds on prior work (doi:10.1038/s41586-021-03819-2)."
    assets = detector.detect_from_text(text)

    assert len(assets['citations']) == 1
    assert '10.1038' in assets['citations'][0].identifier


def test_detect_doi_url():
    """Test DOI URL detection."""
    detector = AssetDetector()

    text = "See https://doi.org/10.1109/CVPR52688.2022.00464 for details."
    assets = detector.detect_from_text(text)

    assert len(assets['citations']) == 1
    assert '10.1109' in assets['citations'][0].identifier


def test_detect_multiple_assets():
    """Test detecting multiple asset types."""
    detector = AssetDetector()

    text = """
    We used the UNI model (https://github.com/mahmoodlab/UNI)
    and compared against BERT. Our method is described in
    doi:10.1000/test.12345.
    """

    assets = detector.detect_from_text(text)

    assert len(assets['github']) >= 1
    assert len(assets['huggingface']) >= 1
    assert len(assets['citations']) >= 1


def test_deduplication():
    """Test that duplicate assets are deduplicated."""
    detector = AssetDetector()

    text = """
    github.com/owner/repo is great!
    Visit https://github.com/owner/repo for more.
    """

    assets = detector.detect_from_text(text)

    # Should only have one entry despite multiple mentions
    assert len(assets['github']) == 1


def test_detect_all_with_passages():
    """Test detect_all with passage list."""
    detector = AssetDetector()

    passages = [
        {'passage_id': 'p1', 'passage_text': 'Code at github.com/owner/repo1'},
        {'passage_id': 'p2', 'passage_text': 'Code at github.com/owner/repo2'},
    ]

    assets = detector.detect_all(passages)

    assert len(assets['github']) == 2


def test_context_extraction():
    """Test that context is extracted around assets."""
    detector = AssetDetector()

    text = "Before. " * 50 + "github.com/owner/repo" + " After." * 50
    assets = detector.detect_from_text(text)

    assert len(assets['github']) == 1
    context = assets['github'][0].context
    assert 'Before' in context
    assert 'After' in context


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
