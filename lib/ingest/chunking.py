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

        # Handle single paragraph larger than max_size - split by sentences/words
        if para_size > max_size:
            # First save any current chunk
            if current_chunk:
                content = '\n\n'.join(current_chunk)
                chunks.append({
                    'content': content,
                    'header': None,
                    'char_start': char_pos - len(content),
                    'char_end': char_pos
                })
                current_chunk = []
                current_size = 0

            # Split large paragraph by sentences or word boundaries
            sub_chunks = _split_large_text(para, max_size, overlap)
            for sub in sub_chunks:
                sub['char_start'] = char_pos + sub['char_start']
                sub['char_end'] = char_pos + sub['char_end']
                chunks.append(sub)
            char_pos += para_size + 2
            continue

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


def _split_large_text(text: str, max_size: int, overlap: int) -> List[Dict]:
    """Split a large text block by sentence or word boundaries."""
    chunks = []

    # Try splitting by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current = []
    current_size = 0
    char_pos = 0

    for sent in sentences:
        sent_size = len(sent)

        # If single sentence is too large, split by words
        if sent_size > max_size:
            if current:
                content = ' '.join(current)
                chunks.append({
                    'content': content,
                    'header': None,
                    'char_start': char_pos - current_size,
                    'char_end': char_pos
                })
                current = []
                current_size = 0

            # Split by words
            words = sent.split()
            word_chunk = []
            word_size = 0
            for word in words:
                if word_size + len(word) + 1 > max_size and word_chunk:
                    content = ' '.join(word_chunk)
                    chunks.append({
                        'content': content,
                        'header': None,
                        'char_start': char_pos,
                        'char_end': char_pos + len(content)
                    })
                    char_pos += len(content) + 1
                    # Overlap
                    if overlap > 0:
                        overlap_words = []
                        overlap_size = 0
                        for w in reversed(word_chunk):
                            if overlap_size + len(w) + 1 <= overlap:
                                overlap_words.insert(0, w)
                                overlap_size += len(w) + 1
                            else:
                                break
                        word_chunk = overlap_words
                        word_size = overlap_size
                    else:
                        word_chunk = []
                        word_size = 0
                word_chunk.append(word)
                word_size += len(word) + 1

            if word_chunk:
                content = ' '.join(word_chunk)
                chunks.append({
                    'content': content,
                    'header': None,
                    'char_start': char_pos,
                    'char_end': char_pos + len(content)
                })
                char_pos += len(content) + 1
            continue

        if current_size + sent_size + 1 > max_size and current:
            content = ' '.join(current)
            chunks.append({
                'content': content,
                'header': None,
                'char_start': char_pos - current_size,
                'char_end': char_pos
            })
            # Overlap from last sentence
            if overlap > 0 and current:
                last = current[-1]
                current = [last[-overlap:]] if len(last) > overlap else [last]
                current_size = len(current[0])
            else:
                current = []
                current_size = 0

        current.append(sent)
        current_size += sent_size + 1
        char_pos = current_size

    if current:
        content = ' '.join(current)
        chunks.append({
            'content': content,
            'header': None,
            'char_start': char_pos - current_size,
            'char_end': char_pos
        })

    return chunks


def estimate_tokens(text: str) -> int:
    """Rough token estimate (4 chars per token)."""
    return len(text) // 4
