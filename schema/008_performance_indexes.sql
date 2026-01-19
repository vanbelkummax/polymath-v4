-- Polymath v4 Schema: Performance Indexes
-- Run: psql -U polymath -d polymath -f schema/008_performance_indexes.sql

-- Partial index for active passages (used in all search queries)
-- Search queries filter by is_superseded = FALSE, so a partial index helps
CREATE INDEX IF NOT EXISTS idx_passages_active
    ON passages(doc_id)
    WHERE is_superseded = FALSE;

-- Covering index for passage search with embedding
-- Optimizes vector search with is_superseded filter
CREATE INDEX IF NOT EXISTS idx_passages_active_embedding
    ON passages USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
    WHERE is_superseded = FALSE AND embedding IS NOT NULL;

-- Index for recent documents (useful for dashboard queries)
CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON documents(created_at DESC);

-- Index for passage concept lookups with high confidence
CREATE INDEX IF NOT EXISTS idx_pc_high_confidence
    ON passage_concepts(passage_id, concept_name)
    WHERE confidence > 0.7;
