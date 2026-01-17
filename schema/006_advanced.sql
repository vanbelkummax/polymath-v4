-- Polymath v4 Schema: Advanced Features
-- Document deduplication, concept normalization, evidence tracking

-- Document aliases for deduplication
CREATE TABLE IF NOT EXISTS doc_aliases (
    alias_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    alias_type TEXT NOT NULL,  -- 'doi', 'pmid', 'arxiv', 'title_hash'
    alias_value TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(alias_type, alias_value)
);

CREATE INDEX IF NOT EXISTS idx_doc_aliases_doc_id ON doc_aliases(doc_id);
CREATE INDEX IF NOT EXISTS idx_doc_aliases_value ON doc_aliases(alias_value);

-- Normalized concept catalog (for consistent tagging)
CREATE TABLE IF NOT EXISTS concepts (
    concept_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    canonical_name TEXT,  -- Normalized form
    concept_type TEXT,    -- method, entity, domain, problem, dataset
    description TEXT,
    parent_concept_id UUID REFERENCES concepts(concept_id),
    is_curated BOOLEAN DEFAULT FALSE,
    occurrence_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_concepts_type ON concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_concepts_parent ON concepts(parent_concept_id);

-- Evidence spans for claim verification
CREATE TABLE IF NOT EXISTS evidence_spans (
    span_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    passage_id UUID NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,
    claim_text TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    span_start INTEGER,
    span_end INTEGER,
    confidence REAL DEFAULT 0.0,
    source_type TEXT,  -- 'paper', 'code', 'manual'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_spans_passage ON evidence_spans(passage_id);

-- Ingest run tracking for auditing
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_type TEXT NOT NULL,  -- 'pdf', 'github', 'hf', 'batch'
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT DEFAULT 'running',  -- running, completed, failed
    items_processed INTEGER DEFAULT 0,
    items_failed INTEGER DEFAULT 0,
    metadata JSONB,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_ingest_runs_status ON ingest_runs(status);
CREATE INDEX IF NOT EXISTS idx_ingest_runs_type ON ingest_runs(run_type);

-- Cross-references between papers and code
CREATE TABLE IF NOT EXISTS code_references (
    ref_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    passage_id UUID REFERENCES passages(passage_id) ON DELETE SET NULL,
    chunk_id UUID REFERENCES code_chunks(chunk_id) ON DELETE SET NULL,
    reference_type TEXT,  -- 'implements', 'cites', 'uses', 'similar'
    confidence REAL DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT ref_has_source CHECK (passage_id IS NOT NULL OR chunk_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS idx_code_refs_passage ON code_references(passage_id);
CREATE INDEX IF NOT EXISTS idx_code_refs_chunk ON code_references(chunk_id);

-- Update concept occurrence count trigger
CREATE OR REPLACE FUNCTION update_concept_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE concepts SET occurrence_count = occurrence_count + 1
    WHERE name = NEW.concept_name;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_concept_count ON passage_concepts;
CREATE TRIGGER trg_update_concept_count
    AFTER INSERT ON passage_concepts
    FOR EACH ROW
    EXECUTE FUNCTION update_concept_count();

-- View: Documents with concept summary
CREATE OR REPLACE VIEW v_document_concepts AS
SELECT
    d.doc_id,
    d.title,
    d.year,
    COUNT(DISTINCT pc.concept_name) as concept_count,
    ARRAY_AGG(DISTINCT pc.concept_type) as concept_types,
    ARRAY_AGG(DISTINCT pc.concept_name) FILTER (WHERE pc.confidence > 0.8) as top_concepts
FROM documents d
LEFT JOIN passages p ON d.doc_id = p.doc_id
LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
GROUP BY d.doc_id, d.title, d.year;

-- View: GitHub repos with paper links
CREATE OR REPLACE VIEW v_repo_paper_links AS
SELECT
    rq.repo_url,
    rq.repo_name,
    rq.status,
    d.title as source_paper,
    d.doc_id as source_doc_id,
    COUNT(DISTINCT cf.file_id) as file_count,
    COUNT(DISTINCT cc.chunk_id) as chunk_count
FROM repo_queue rq
LEFT JOIN paper_repos pr ON rq.repo_url = pr.repo_url
LEFT JOIN documents d ON pr.doc_id = d.doc_id
LEFT JOIN code_files cf ON rq.repo_name = cf.repo_name
LEFT JOIN code_chunks cc ON cf.file_id = cc.file_id
GROUP BY rq.queue_id, rq.repo_url, rq.repo_name, rq.status, d.title, d.doc_id;
