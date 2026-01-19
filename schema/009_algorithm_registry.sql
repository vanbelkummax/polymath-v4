-- Algorithm Registry Schema
-- Comprehensive database of algorithms with cross-domain applications
-- Created: 2026-01-19

-- Main algorithms table
CREATE TABLE IF NOT EXISTS algorithms (
    algo_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL UNIQUE,
    canonical_name VARCHAR(255),  -- Standardized name
    aliases TEXT[],               -- Alternative names

    -- Classification
    original_domain VARCHAR(100), -- "topology", "control_theory", "optimization", etc.
    category VARCHAR(100),        -- "clustering", "decomposition", "transform", etc.
    subcategory VARCHAR(100),

    -- Description
    description TEXT,
    what_it_does TEXT,            -- Plain English explanation
    mathematical_formulation TEXT, -- LaTeX or description

    -- Complexity
    time_complexity VARCHAR(50),  -- O(n), O(n²), etc.
    space_complexity VARCHAR(50),

    -- Applications
    traditional_uses TEXT[],      -- Original domain applications
    spatial_biology_uses TEXT[],  -- Specific to our domain
    polymathic_potential TEXT,    -- Cross-domain transfer notes

    -- Quality flags
    ocr_quality_flag VARCHAR(20) DEFAULT 'unknown', -- 'good', 'suspect', 'needs_review'
    ocr_quality_notes TEXT,
    verified BOOLEAN DEFAULT FALSE,
    verification_date TIMESTAMP,

    -- Metadata
    mention_count INTEGER DEFAULT 0,
    first_seen_paper_id UUID REFERENCES documents(doc_id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Algorithm to paper links (evidence)
CREATE TABLE IF NOT EXISTS algorithm_papers (
    algo_id INTEGER REFERENCES algorithms(algo_id) ON DELETE CASCADE,
    doc_id UUID REFERENCES documents(doc_id) ON DELETE CASCADE,
    passage_ids UUID[],           -- Specific passages mentioning this algorithm
    context_type VARCHAR(50),     -- 'definition', 'application', 'comparison', 'citation'
    relevance_score FLOAT,
    PRIMARY KEY (algo_id, doc_id)
);

-- Algorithm to repository links (implementations)
CREATE TABLE IF NOT EXISTS algorithm_repos (
    algo_id INTEGER REFERENCES algorithms(algo_id) ON DELETE CASCADE,
    repo_id UUID REFERENCES repositories(repo_id) ON DELETE CASCADE,
    implementation_quality VARCHAR(20), -- 'reference', 'production', 'educational', 'unknown'
    language VARCHAR(50),
    file_paths TEXT[],            -- Specific files implementing the algorithm
    PRIMARY KEY (algo_id, repo_id)
);

-- Cross-domain bridges (polymathic transfers)
CREATE TABLE IF NOT EXISTS algorithm_bridges (
    bridge_id SERIAL PRIMARY KEY,
    algo_id INTEGER REFERENCES algorithms(algo_id) ON DELETE CASCADE,
    source_domain VARCHAR(100) NOT NULL,
    source_application TEXT,
    target_domain VARCHAR(100) NOT NULL,
    target_application TEXT,
    transfer_mechanism TEXT,      -- How the transfer works
    success_evidence TEXT[],      -- Papers/passages showing successful transfer
    difficulty VARCHAR(20),       -- 'direct', 'moderate', 'significant_adaptation'
    polymathic_score FLOAT,       -- 0-1, how novel/valuable is this transfer
    created_at TIMESTAMP DEFAULT NOW()
);

-- Algorithm similarity (for finding related algorithms)
CREATE TABLE IF NOT EXISTS algorithm_similarity (
    algo_id_1 INTEGER REFERENCES algorithms(algo_id) ON DELETE CASCADE,
    algo_id_2 INTEGER REFERENCES algorithms(algo_id) ON DELETE CASCADE,
    similarity_score FLOAT,
    similarity_type VARCHAR(50),  -- 'mathematical', 'application', 'domain', 'embedding'
    PRIMARY KEY (algo_id_1, algo_id_2)
);

-- Domain taxonomy
CREATE TABLE IF NOT EXISTS algorithm_domains (
    domain_id SERIAL PRIMARY KEY,
    domain_name VARCHAR(100) UNIQUE NOT NULL,
    parent_domain VARCHAR(100),
    description TEXT,
    is_polymathic_source BOOLEAN DEFAULT FALSE, -- Rich source for cross-domain transfer
    spatial_relevance VARCHAR(20) -- 'high', 'medium', 'low', 'unexplored'
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_algorithms_domain ON algorithms(original_domain);
CREATE INDEX IF NOT EXISTS idx_algorithms_category ON algorithms(category);
CREATE INDEX IF NOT EXISTS idx_algorithms_name_trgm ON algorithms USING gin(name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_algorithms_verified ON algorithms(verified) WHERE verified = TRUE;
CREATE INDEX IF NOT EXISTS idx_algo_bridges_domains ON algorithm_bridges(source_domain, target_domain);

-- Full text search on algorithm descriptions
CREATE INDEX IF NOT EXISTS idx_algorithms_fts ON algorithms
    USING gin(to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '') || ' ' || COALESCE(what_it_does, '')));

-- Insert base domains
INSERT INTO algorithm_domains (domain_name, parent_domain, description, is_polymathic_source, spatial_relevance) VALUES
    ('topology', NULL, 'Topological methods including TDA, persistent homology', TRUE, 'high'),
    ('optimization', NULL, 'Optimization algorithms and methods', FALSE, 'high'),
    ('linear_algebra', NULL, 'Matrix operations, decompositions, transforms', FALSE, 'high'),
    ('graph_theory', NULL, 'Graph algorithms and network analysis', FALSE, 'high'),
    ('statistics', NULL, 'Statistical methods and inference', FALSE, 'high'),
    ('control_theory', NULL, 'Dynamical systems and control', TRUE, 'medium'),
    ('game_theory', NULL, 'Strategic decision making, equilibria', TRUE, 'medium'),
    ('information_theory', NULL, 'Entropy, mutual information, coding', TRUE, 'high'),
    ('signal_processing', NULL, 'Transforms, filtering, spectral methods', FALSE, 'high'),
    ('machine_learning', NULL, 'Learning algorithms', FALSE, 'high'),
    ('deep_learning', 'machine_learning', 'Neural network methods', FALSE, 'high'),
    ('compressed_sensing', 'signal_processing', 'Sparse reconstruction', TRUE, 'high'),
    ('optimal_transport', 'optimization', 'Wasserstein distances, transport maps', TRUE, 'high'),
    ('category_theory', NULL, 'Abstract algebra, functors, sheaves', TRUE, 'medium'),
    ('differential_geometry', NULL, 'Manifolds, curvature, geodesics', TRUE, 'medium'),
    ('tropical_geometry', NULL, 'Max-plus algebra, tropical methods', TRUE, 'low'),
    ('renormalization', 'physics', 'Multi-scale coarse graining', TRUE, 'medium')
ON CONFLICT (domain_name) DO NOTHING;

COMMENT ON TABLE algorithms IS 'Comprehensive registry of algorithms with cross-domain applications for spatial biology';
COMMENT ON TABLE algorithm_bridges IS 'Polymathic transfers - algorithms applied across domains';
COMMENT ON COLUMN algorithms.ocr_quality_flag IS 'Flag for math OCR quality: good, suspect, needs_review';
