-- Polymath v4 Code & Assets Schema
-- Run: psql -U polymath -d polymath -f schema/003_code.sql

-- GitHub repository queue
CREATE TABLE IF NOT EXISTS repo_queue (
    queue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url TEXT UNIQUE NOT NULL,
    owner TEXT,
    repo_name TEXT,

    -- Priority (higher = more papers cite it)
    priority INTEGER DEFAULT 5,
    source_doc_count INTEGER DEFAULT 1,

    -- Status
    status TEXT DEFAULT 'pending',  -- pending, cloning, ingesting, complete, failed

    -- Provenance
    first_seen_doc_id UUID REFERENCES documents(doc_id),
    source TEXT,  -- paper_detection, manual, discovery, user:username
    context TEXT,  -- Surrounding text where found

    -- Ingestion
    local_path TEXT,
    files_count INTEGER,
    chunks_count INTEGER,
    ingested_at TIMESTAMPTZ,
    error_message TEXT,

    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Link papers to repos
CREATE TABLE IF NOT EXISTS paper_repos (
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_url TEXT,
    passage_id UUID,
    context TEXT,
    PRIMARY KEY (doc_id, repo_url)
);

-- Code files
CREATE TABLE IF NOT EXISTS code_files (
    file_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_name TEXT NOT NULL,
    repo_url TEXT,
    repo_root TEXT,
    default_branch TEXT,
    head_commit_sha TEXT,

    file_path TEXT NOT NULL,
    language TEXT,
    file_hash TEXT,
    file_size_bytes INTEGER,
    loc INTEGER,
    is_generated BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(repo_name, file_path, head_commit_sha)
);

-- Code chunks (functions, classes, methods)
CREATE TABLE IF NOT EXISTS code_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id UUID REFERENCES code_files(file_id) ON DELETE CASCADE,

    chunk_type TEXT NOT NULL,  -- function, method, class, module
    name TEXT,
    class_name TEXT,
    symbol_qualified_name TEXT,

    start_line INTEGER,
    end_line INTEGER,
    content TEXT NOT NULL,
    chunk_hash TEXT,

    docstring TEXT,
    signature TEXT,
    imports TEXT[],
    concepts TEXT[],  -- Auto-detected concepts

    embedding_id TEXT,  -- For ChromaDB lookup

    created_at TIMESTAMPTZ DEFAULT now()
);

-- HuggingFace models
CREATE TABLE IF NOT EXISTS hf_models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT,
    organization TEXT,
    pipeline_tag TEXT,
    library_name TEXT,

    -- Metadata
    architectures TEXT[],
    tags TEXT[],
    downloads_30d INTEGER,
    likes INTEGER,
    model_card_summary TEXT,

    -- Tracking
    first_seen_doc_id UUID REFERENCES documents(doc_id),
    citation_count INTEGER DEFAULT 1,
    last_metadata_fetch TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Unresolved HF model mentions
CREATE TABLE IF NOT EXISTS hf_model_mentions (
    mention_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(doc_id),
    model_id_raw TEXT NOT NULL,  -- As found in text
    passage_id UUID,
    context TEXT,

    resolved BOOLEAN DEFAULT FALSE,
    resolved_to_model_id TEXT REFERENCES hf_models(model_id),

    created_at TIMESTAMPTZ DEFAULT now()
);

-- Link papers to HF models
CREATE TABLE IF NOT EXISTS paper_hf_models (
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    model_id TEXT,
    passage_id UUID,
    context TEXT,
    PRIMARY KEY (doc_id, model_id)
);

-- Citation links
CREATE TABLE IF NOT EXISTS citation_links (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    citing_doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    cited_doc_id UUID REFERENCES documents(doc_id),  -- NULL if not in corpus
    cited_doi TEXT,
    citation_context TEXT,
    in_corpus BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(citing_doc_id, cited_doi)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_repo_queue_status ON repo_queue(status);
CREATE INDEX IF NOT EXISTS idx_repo_queue_priority ON repo_queue(priority DESC);
CREATE INDEX IF NOT EXISTS idx_code_files_repo ON code_files(repo_name);
CREATE INDEX IF NOT EXISTS idx_code_chunks_file ON code_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_code_chunks_type ON code_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_code_chunks_name ON code_chunks(name);

-- Views
CREATE OR REPLACE VIEW v_repo_queue_summary AS
SELECT
    status,
    COUNT(*) as count,
    ROUND(AVG(priority), 1) as avg_priority,
    SUM(files_count) as total_files,
    SUM(chunks_count) as total_chunks
FROM repo_queue
GROUP BY status;

CREATE OR REPLACE VIEW v_top_repos AS
SELECT
    repo_url,
    owner,
    repo_name,
    source_doc_count,
    priority,
    status,
    files_count,
    chunks_count
FROM repo_queue
ORDER BY source_doc_count DESC, priority DESC
LIMIT 50;
