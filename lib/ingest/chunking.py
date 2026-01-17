"""
Text Chunking for Polymath v4

Splits text into semantically meaningful chunks for embedding.
"""

import re
from typing import List, Dict


def chunk_text(
    text: str,
    max_size: int = 1500,
    overlap: int = 200
) -> List[Dict]:
    """
    Chunk text by headers or fixed size.

    Returns:
        List of dicts with 'content', 'header', 'char_start', 'char_end'
    """
    # Try header-based chunking first
    if '##' in text or '\n# ' in text:
        chunks = chunk_by_headers(text, max_size)
        if chunks:
            return chunks

    # Fall back to fixed-size chunking
    return chunk_fixed_size(text, max_size, overlap)


def chunk_by_headers(text: str, max_size: int = 1500) -> List[Dict]:
    """Split text on markdown headers."""
    # Pattern for headers
    pattern = r'^(#{1,3})\s+(.+?)$'
    chunks = []

    # Find all headers
    headers = list(re.finditer(pattern, text, re.MULTILINE))

    if not headers:
        return []

    for i, match in enumerate(headers):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)

        content = text[start:end].strip()

        if len(content) < 50:
            continue

        # Split large sections
        if len(content) > max_size:
            sub_chunks = chunk_fixed_size(content, max_size, 100)
            for j, sub in enumerate(sub_chunks):
                sub['header'] = f"{title} (part {j+1})"
                chunks.append(sub)
        else:
            chunks.append({
                'content': content,
                'header': title,
                'char_start': start,
                'char_end': end
            })

    return chunks


def chunk_fixed_size(
    text: str,
    max_size: int = 1500,
    overlap: int = 200
) -> List[Dict]:
    """Split text into fixed-size chunks with overlap."""
    chunks = []

    # Split on paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)

    current_chunk = []
    current_size = 0
    char_pos = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_size = len(para)

        if current_size + para_size > max_size and current_chunk:
            # Save current chunk
            content = '\n\n'.join(current_chunk)
            chunks.append({
                'content': content,
                'header': None,
                'char_start': char_pos - len(content),
                'char_end': char_pos
            })

            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-1][-overlap:] if len(current_chunk[-1]) > overlap else current_chunk[-1]
                current_chunk = [overlap_text]
                current_size = len(overlap_text)
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(para)
        current_size += para_size
        char_pos += para_size + 2  # +2 for \n\n

    # Don't forget last chunk
    if current_chunk:
        content = '\n\n'.join(current_chunk)
        chunks.append({
            'content': content,
            'header': None,
            'char_start': char_pos - len(content),
            'char_end': char_pos
        })

    return chunks


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4
