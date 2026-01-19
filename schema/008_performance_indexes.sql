-- Polymath v4 Schema: Performance Indexes
-- Run: psql -U polymath -d polymath -f schema/008_performance_indexes.sql

-- Partial index for active passages (used in all search queries)
-- Search queries filter by is_superseded = FALSE, so a partial index helps
CREATE INDEX IF NOT EXISTS idx_passages_active
    ON passages(doc_id)
    WHERE is_superseded = FALSE;

-- Vector search index using HNSW (more accurate than IVFFlat)
-- HNSW provides better recall than IVFFlat for approximate nearest neighbor search
-- m=16: connections per node, ef_construction=200: build-time quality
-- NOTE: Do NOT use partial indexes with HNSW - they cause incorrect results
-- when the query planner chooses the index over sequential scan
CREATE INDEX IF NOT EXISTS idx_passages_embedding_hnsw
    ON passages USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Index for recent documents (useful for dashboard queries)
CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON documents(created_at DESC);

-- Index for passage concept lookups with high confidence
CREATE INDEX IF NOT EXISTS idx_pc_high_confidence
    ON passage_concepts(passage_id, concept_name)
    WHERE confidence > 0.7;
