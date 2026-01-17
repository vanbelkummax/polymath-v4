-- Polymath v4 Concepts Schema
-- Run: psql -U polymath -d polymath -f schema/002_concepts.sql

-- Passage concepts
CREATE TABLE IF NOT EXISTS passage_concepts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    passage_id UUID NOT NULL REFERENCES passages(passage_id) ON DELETE CASCADE,

    -- Concept
    concept_name TEXT NOT NULL,
    concept_type TEXT NOT NULL,  -- method, problem, domain, dataset, metric, entity
    confidence FLOAT DEFAULT 0.8,
    evidence TEXT,  -- Quote from source

    -- Extraction metadata
    extractor_model TEXT,
    extractor_version TEXT,

    created_at TIMESTAMPTZ DEFAULT now(),

    UNIQUE(passage_id, concept_name)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_pc_passage ON passage_concepts(passage_id);
CREATE INDEX IF NOT EXISTS idx_pc_concept ON passage_concepts(concept_name);
CREATE INDEX IF NOT EXISTS idx_pc_type ON passage_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_pc_name_trgm ON passage_concepts USING gin(concept_name gin_trgm_ops);

-- Concept type enum check
ALTER TABLE passage_concepts
    DROP CONSTRAINT IF EXISTS check_concept_type;
ALTER TABLE passage_concepts
    ADD CONSTRAINT check_concept_type
    CHECK (concept_type IN ('method', 'problem', 'domain', 'dataset', 'metric', 'entity', 'tool'));

-- Batch job tracking for async concept extraction
CREATE TABLE IF NOT EXISTS concept_batch_jobs (
    job_id TEXT PRIMARY KEY,
    gcs_input_uri TEXT,
    gcs_output_uri TEXT,
    passage_count INTEGER,
    status TEXT DEFAULT 'pending',  -- pending, running, succeeded, failed
    submitted_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    error_message TEXT
);

-- View: Concept frequency
CREATE OR REPLACE VIEW v_concept_frequency AS
SELECT
    concept_name,
    concept_type,
    COUNT(*) as frequency,
    COUNT(DISTINCT pc.passage_id) as passage_count,
    ROUND(AVG(confidence)::numeric, 2) as avg_confidence
FROM passage_concepts pc
GROUP BY concept_name, concept_type
ORDER BY frequency DESC;

-- View: Document concept summary
CREATE OR REPLACE VIEW v_document_concepts AS
SELECT
    d.doc_id,
    d.title,
    COUNT(DISTINCT pc.concept_name) as unique_concepts,
    COUNT(DISTINCT CASE WHEN pc.concept_type = 'method' THEN pc.concept_name END) as method_count,
    COUNT(DISTINCT CASE WHEN pc.concept_type = 'domain' THEN pc.concept_name END) as domain_count
FROM documents d
LEFT JOIN passages p ON d.doc_id = p.doc_id
LEFT JOIN passage_concepts pc ON p.passage_id = pc.passage_id
GROUP BY d.doc_id, d.title;
