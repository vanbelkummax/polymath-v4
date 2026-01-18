# Polymath v4 Fresh Ingestion Pipeline Design

**Date:** 2026-01-17
**Status:** Approved, ready for implementation

---

## Goals

Build a polymathic knowledge base supporting:
1. **Research assistant** - semantic search during reading/writing
2. **Literature review** - synthesis across topics (metadata richness)
3. **Cross-domain connections** - concept extraction and bridging
4. **Coding agent** - GitHub repos extracted from papers

---

## Architecture: Three-Stage Pipeline

### Stage 1: INGEST (Fast, ~4 hours for 1,400 PDFs)

```
Zotero CSV (5,348 entries)
    ↓
Deduplicate (→ ~4,500 unique)
    ↓
Match PDFs to Zotero metadata
    ↓
For each PDF:
    ├── Parse with LaTeX preservation
    ├── Chunk text (header-aware)
    ├── Compute BGE-M3 embeddings
    ├── Detect assets (GitHub, HF, DOIs)
    └── Store in Postgres
```

**Output:** Working search system with basic metadata

### Stage 2: ENRICH (Background, incremental)

```
For docs with DOI:
    ├── CrossRef lookup (title, authors, venue, year)
    ├── OpenAlex lookup (citations, concepts, open access)
    └── Semantic Scholar (citation counts, influential citations)

For docs without DOI:
    ├── Parse authors from PDF header
    ├── Extract year from text
    └── Fuzzy match against OpenAlex
```

**Output:** Rich metadata, citation counts, venue info

### Stage 3: CONCEPTS (Batch job via Gemini)

```
For passages without concepts:
    ├── Submit to Gemini batch API
    ├── Extract concepts with types and confidence
    └── Sync to Neo4j graph
```

**Output:** Cross-domain concept graph

---

## Deduplication Strategy

**Approach:** Strict (prefer duplicates over lost papers)

### Dedup Keys (in order of precedence)

1. **DOI match** - if DOI exists in both, definite duplicate
2. **PDF hash match** - same file content, definite duplicate
3. **Title hash + year + author overlap** - all three must match

### What counts as "different papers"

- Same title, different DOI → different papers (ingest both)
- Same title, different authors → different papers
- Same title, different year → different papers
- Same title, same year, same authors, different PDF hash → flag for review

### Implementation

```python
def is_duplicate(new_doc, existing_docs):
    # 1. DOI match (definitive)
    if new_doc.doi and any(d.doi == new_doc.doi for d in existing_docs):
        return True

    # 2. PDF hash match (same file)
    if new_doc.pdf_hash and any(d.pdf_hash == new_doc.pdf_hash for d in existing_docs):
        return True

    # 3. Title + year + authors (strict)
    for d in existing_docs:
        if (d.title_hash == new_doc.title_hash and
            d.year == new_doc.year and
            authors_overlap(d.authors, new_doc.authors) > 0.5):
            return True

    return False
```

---

## PDF Parsing Enhancements

### LaTeX Preservation

- Use PyMuPDF with `preserve_whitespace=True`
- Detect and preserve Unicode math symbols (∑, ∫, √, etc.)
- Keep LaTeX source when embedded (common in arXiv PDFs)
- Flag `has_equations=True` for later Nougat enrichment

### OCR Fallback

- Enable for PDFs with <100 chars extracted
- Use pytesseract at 300 DPI
- Mark `extraction_method='ocr'` for quality tracking

### Quality Indicators

```python
@dataclass
class ParseResult:
    text: str
    has_text: bool
    is_scanned: bool
    has_equations: bool
    extraction_method: str  # 'fitz', 'ocr', 'fitz+latex'
    quality_score: float    # 0-1 based on heuristics
```

---

## Zotero Integration

### CSV Fields Used

| Field | Maps to | Notes |
|-------|---------|-------|
| Key | zotero_key | Unique identifier |
| DOI | doi | Primary dedup key |
| Title | title | Normalized for title_hash |
| Author | authors | Parsed to array |
| Publication Year | year | Integer |
| Abstract Note | abstract | Full abstract |
| Publication Title | venue | Journal/conference |
| File Attachments | pdf_path | Windows path, needs mapping |

### Path Mapping

```python
# Zotero stores Windows paths, need to map to Linux
ZOTERO_PATH_MAP = {
    r"C:\Users\User\Zotero\storage": "/mnt/c/Users/User/Zotero/storage",
}
```

---

## Execution Plan

### Tonight (Stage 1)

```bash
cd /home/user/polymath-v4

# 1. Deduplicate and prepare Zotero CSV
python scripts/prepare_zotero_ingest.py '/mnt/c/Users/User/Downloads/My Library.csv' \
    --output /home/user/work/polymax/ingest_staging/zotero_deduped.csv

# 2. Run batch ingest (background, ~4 hours)
nohup python scripts/ingest_pdf.py /home/user/work/polymax/ingest_staging/ \
    --workers 2 \
    --zotero-csv /home/user/work/polymax/ingest_staging/zotero_deduped.csv \
    --batch-name fresh_build_2026_01_17 \
    > /home/user/logs/ingest_fresh_build.log 2>&1 &

# 3. Monitor progress
tail -f /home/user/logs/ingest_fresh_build.log
```

### Tomorrow (Stage 2)

```bash
# Run metadata enrichment
python scripts/enrich_metadata.py --source crossref --limit 500
python scripts/enrich_metadata.py --source openalex --limit 500
```

### Later (Stage 3)

```bash
# Submit concept extraction batch
python scripts/batch_concepts.py --submit --limit 1000
```

---

## Database Changes

### New columns needed

```sql
-- Add to documents table
ALTER TABLE documents ADD COLUMN IF NOT EXISTS pdf_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_pdf_hash ON documents(pdf_hash) WHERE pdf_hash IS NOT NULL;

-- Track extraction quality
ALTER TABLE documents ADD COLUMN IF NOT EXISTS extraction_method TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS extraction_quality REAL;
```

### Preserve existing data

- Keep all `paper_repos` (1,603 repos)
- Keep all `passage_concepts` (4.8M concepts)
- New docs added alongside, dedup prevents true duplicates

---

## Skills to Create/Update

| Skill | Purpose |
|-------|---------|
| `polymath-fresh-ingest` | Run the full Stage 1 pipeline |
| `polymath-zotero-prep` | Deduplicate and prepare Zotero CSV |
| `polymath-metadata-enrich` | Run Stage 2 enrichment |

---

## Success Criteria

1. **Search works** - `search("spatial transcriptomics")` returns relevant papers
2. **No true duplicates** - same DOI appears only once
3. **Metadata quality** - >80% of papers have DOI, >60% have authors
4. **Repos preserved** - 1,603+ repos in paper_repos
5. **Concepts linkable** - existing concepts can be re-linked after ingest

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Zotero paths don't map | Build path mapper with fallback to filename match |
| OCR too slow | Disable OCR for first pass, flag scanned PDFs for later |
| Memory issues with 1,400 PDFs | Use --workers 2, process in batches |
| Duplicates slip through | Post-ingest dedup report, manual review queue |
