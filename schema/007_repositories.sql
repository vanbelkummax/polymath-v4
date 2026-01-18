-- 007_repositories.sql
-- Add GitHub repositories as first-class entities
-- Date: 2026-01-18

-- Core repository table
CREATE TABLE IF NOT EXISTS repositories (
    repo_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_url TEXT UNIQUE NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,

    -- GitHub metadata
    description TEXT,
    language TEXT,
    stars INT,
    forks INT,
    topics TEXT[],
    default_branch TEXT DEFAULT 'main',

    -- Content
    readme_content TEXT,

    -- Tracking
    source_method TEXT,  -- 'paper_detection', 'curated', 'orphaned'
    github_id BIGINT,
    indexed_at TIMESTAMP DEFAULT NOW(),
    last_github_sync TIMESTAMP,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Repo passages (parallel to paper passages)
CREATE TABLE IF NOT EXISTS repo_passages (
    passage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id UUID NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,

    passage_text TEXT NOT NULL,
    section TEXT,           -- 'readme', 'docstring', 'module_doc', 'class_doc'
    file_path TEXT,         -- e.g., 'src/analysis.py'
    function_name TEXT,     -- e.g., 'calculate_moran'
    class_name TEXT,        -- e.g., 'SpatialAnalyzer'

    embedding vector(1024),
    quality_score REAL,

    created_at TIMESTAMP DEFAULT NOW()
);

-- Paper-repo links (bidirectional, replaces paper_repos for linking)
CREATE TABLE IF NOT EXISTS paper_repo_links (
    doc_id UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    repo_id UUID NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,

    link_type TEXT DEFAULT 'mentioned',  -- 'mentioned', 'implements', 'uses', 'cites'
    confidence REAL DEFAULT 1.0,
    context_snippet TEXT,  -- text around the mention
    detected_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (doc_id, repo_id)
);

-- Repo concepts (parallel to passage_concepts)
CREATE TABLE IF NOT EXISTS repo_concepts (
    passage_id UUID NOT NULL REFERENCES repo_passages(passage_id) ON DELETE CASCADE,
    concept_name TEXT NOT NULL,
    concept_type TEXT,  -- 'method', 'algorithm', 'library', 'data_structure'
    confidence REAL DEFAULT 1.0,
    extractor_version TEXT,
    created_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (passage_id, concept_name)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_repositories_owner_name ON repositories(owner, name);
CREATE INDEX IF NOT EXISTS idx_repositories_language ON repositories(language);
CREATE INDEX IF NOT EXISTS idx_repositories_stars ON repositories(stars DESC);

CREATE INDEX IF NOT EXISTS idx_repo_passages_repo_id ON repo_passages(repo_id);
CREATE INDEX IF NOT EXISTS idx_repo_passages_section ON repo_passages(section);

CREATE INDEX IF NOT EXISTS idx_paper_repo_links_repo ON paper_repo_links(repo_id);
CREATE INDEX IF NOT EXISTS idx_paper_repo_links_doc ON paper_repo_links(doc_id);

-- Vector index for repo passage embeddings
CREATE INDEX IF NOT EXISTS idx_repo_passages_embedding ON repo_passages
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- View for unified search across papers and repos
CREATE OR REPLACE VIEW all_passages AS
SELECT
    passage_id,
    doc_id AS source_id,
    'paper' AS source_type,
    passage_text,
    section,
    embedding
FROM passages
WHERE embedding IS NOT NULL
UNION ALL
SELECT
    passage_id,
    repo_id AS source_id,
    'repo' AS source_type,
    passage_text,
    section,
    embedding
FROM repo_passages
WHERE embedding IS NOT NULL;

COMMENT ON TABLE repositories IS 'GitHub repositories indexed for content search';
COMMENT ON TABLE repo_passages IS 'Text passages from repo READMEs and docstrings';
COMMENT ON TABLE paper_repo_links IS 'Links between papers and repositories they mention';
COMMENT ON VIEW all_passages IS 'Unified view of paper and repo passages for search';
