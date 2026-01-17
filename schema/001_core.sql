-- Polymath v4 Core Schema
-- Run: psql -U polymath -d polymath -f schema/001_core.sql

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    title_hash TEXT,  -- For deduplication
    authors TEXT[],
    year INTEGER,
    venue TEXT,

    -- Identifiers
    doi TEXT UNIQUE,
    pmid TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    zotero_key TEXT,

    -- Content
    abstract TEXT,
    pdf_path TEXT,

    -- Metadata
    source_method TEXT,  -- How metadata was resolved
    metadata_confidence FLOAT DEFAULT 0.5,
    ingest_batch TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Passages table with vector embeddings
CREATE TABLE IF NOT EXISTS passages (
    passage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,

    -- Content
    passage_text TEXT NOT NULL,
    section TEXT,
    parent_section TEXT,

    -- Position
    page_num INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    passage_index INTEGER,

    -- Embedding (1024-dim for BGE-M3)
    embedding vector(1024),

    -- Lifecycle
    is_superseded BOOLEAN DEFAULT FALSE,
    superseded_at TIMESTAMPTZ,
    superseded_by_batch TEXT,

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_passages_doc_id ON passages(doc_id);
CREATE INDEX IF NOT EXISTS idx_passages_embedding ON passages USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_documents_title_hash ON documents(title_hash);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi) WHERE doi IS NOT NULL;

-- Full-text search
CREATE INDEX IF NOT EXISTS idx_passages_fts ON passages USING gin(to_tsvector('english', passage_text));

-- Ingest batch tracking
CREATE TABLE IF NOT EXISTS ingest_batches (
    batch_name TEXT PRIMARY KEY,
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    succeeded_files INTEGER DEFAULT 0,
    failed_files INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',
    started_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- Helper function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
