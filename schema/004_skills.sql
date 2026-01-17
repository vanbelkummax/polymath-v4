-- Polymath v4 Skills Schema
-- Run: psql -U polymath -d polymath -f schema/004_skills.sql

-- Paper-derived skills
CREATE TABLE IF NOT EXISTS paper_skills (
    skill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name TEXT UNIQUE NOT NULL,
    skill_type TEXT,  -- method, workflow, analysis, integration

    -- Content
    description TEXT,
    prerequisites TEXT[],
    steps TEXT[],
    pitfalls TEXT[],

    -- Embedding for deduplication
    embedding vector(1024),

    -- Status
    status TEXT DEFAULT 'draft',  -- draft, candidate, promoted, deprecated

    -- Evidence tracking
    evidence_count INTEGER DEFAULT 0,
    source_passage_ids UUID[],
    source_doc_ids UUID[],
    code_links TEXT[],  -- GitHub URLs

    -- Promotion
    promoted_at TIMESTAMPTZ,
    promoted_by TEXT,
    oracle_test_path TEXT,

    -- Canonical tracking (for dedup)
    is_canonical BOOLEAN DEFAULT TRUE,
    canonical_skill_id UUID REFERENCES paper_skills(skill_id),
    merge_count INTEGER DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Skill usage tracking (for Gate 4)
CREATE TABLE IF NOT EXISTS skill_usage_log (
    usage_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name TEXT NOT NULL,
    task_description TEXT,
    outcome TEXT NOT NULL,  -- success, failure, partial
    oracle_passed BOOLEAN,
    failure_notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Skill bridges (cross-domain transfers)
CREATE TABLE IF NOT EXISTS skill_bridges (
    bridge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_skill_id UUID REFERENCES paper_skills(skill_id),
    target_skill_id UUID REFERENCES paper_skills(skill_id),
    bridge_type TEXT,  -- analogy, generalization, specialization
    confidence FLOAT DEFAULT 0.5,
    validation_status TEXT DEFAULT 'proposed',  -- proposed, validated, rejected
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_skills_status ON paper_skills(status);
CREATE INDEX IF NOT EXISTS idx_skills_type ON paper_skills(skill_type);
CREATE INDEX IF NOT EXISTS idx_skills_embedding ON paper_skills USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20);
CREATE INDEX IF NOT EXISTS idx_usage_skill ON skill_usage_log(skill_name);
CREATE INDEX IF NOT EXISTS idx_usage_outcome ON skill_usage_log(outcome);

-- Constraints
ALTER TABLE paper_skills
    DROP CONSTRAINT IF EXISTS check_skill_status;
ALTER TABLE paper_skills
    ADD CONSTRAINT check_skill_status
    CHECK (status IN ('draft', 'candidate', 'promoted', 'deprecated'));

ALTER TABLE skill_usage_log
    DROP CONSTRAINT IF EXISTS check_usage_outcome;
ALTER TABLE skill_usage_log
    ADD CONSTRAINT check_usage_outcome
    CHECK (outcome IN ('success', 'failure', 'partial'));

-- Helper function: Log skill usage
CREATE OR REPLACE FUNCTION log_skill_usage(
    p_skill_name TEXT,
    p_outcome TEXT,
    p_oracle_passed BOOLEAN DEFAULT NULL,
    p_task_description TEXT DEFAULT NULL,
    p_failure_notes TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_usage_id UUID;
BEGIN
    INSERT INTO skill_usage_log (skill_name, outcome, oracle_passed, task_description, failure_notes)
    VALUES (p_skill_name, p_outcome, p_oracle_passed, p_task_description, p_failure_notes)
    RETURNING usage_id INTO v_usage_id;

    RETURN v_usage_id;
END;
$$ LANGUAGE plpgsql;

-- Views
CREATE OR REPLACE VIEW v_promotion_candidates AS
SELECT
    s.skill_id,
    s.skill_name,
    s.skill_type,
    s.evidence_count,
    s.description,
    COALESCE(
        (SELECT COUNT(*) FROM skill_usage_log u
         WHERE u.skill_name = s.skill_name AND u.outcome = 'success'),
        0
    ) as success_count,
    CASE
        WHEN s.evidence_count >= 2 THEN 'PASS'
        WHEN s.evidence_count = 1 AND array_length(s.code_links, 1) >= 1 THEN 'PASS'
        ELSE 'FAIL'
    END as gate_1_evidence,
    CASE
        WHEN s.oracle_test_path IS NOT NULL THEN 'PASS'
        ELSE 'MISSING'
    END as gate_2_oracle
FROM paper_skills s
WHERE s.status = 'draft'
ORDER BY s.evidence_count DESC, s.created_at DESC;

CREATE OR REPLACE VIEW v_skill_usage_summary AS
SELECT
    skill_name,
    COUNT(*) as total_uses,
    SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
    SUM(CASE WHEN outcome = 'failure' THEN 1 ELSE 0 END) as failures,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) as success_rate
FROM skill_usage_log
GROUP BY skill_name
ORDER BY total_uses DESC;
