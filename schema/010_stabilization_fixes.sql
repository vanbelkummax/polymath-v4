-- Polymath v4 Schema Migration: Stabilization Fixes
-- Date: 2026-01-19
-- Audit: STABILIZATION_AUDIT_2026_01_19.md
--
-- Run: psql -U polymath -d polymath -f schema/010_stabilization_fixes.sql

-- Fix 1: Add missing evidence_count column to paper_skills
-- Required by: scripts/system_report.py (get_skill_stats function)
ALTER TABLE paper_skills
ADD COLUMN IF NOT EXISTS evidence_count INTEGER DEFAULT 0;

-- Fix 2: Ensure paper_repo_links has proper indexes for code search
CREATE INDEX IF NOT EXISTS idx_paper_repo_links_doc_repo
ON paper_repo_links (doc_id, repo_id);

-- Fix 3: Add index for faster code search queries
CREATE INDEX IF NOT EXISTS idx_documents_doc_id_title
ON documents (doc_id, title);

-- Verification queries
DO $$
BEGIN
    -- Verify evidence_count column exists
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'paper_skills' AND column_name = 'evidence_count'
    ) THEN
        RAISE NOTICE 'evidence_count column: OK';
    ELSE
        RAISE WARNING 'evidence_count column: MISSING';
    END IF;

    -- Verify paper_repo_links index
    IF EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_paper_repo_links_doc_repo'
    ) THEN
        RAISE NOTICE 'idx_paper_repo_links_doc_repo: OK';
    ELSE
        RAISE WARNING 'idx_paper_repo_links_doc_repo: MISSING';
    END IF;
END $$;

-- Migration complete message
DO $$
BEGIN
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Migration 010_stabilization_fixes.sql complete';
    RAISE NOTICE 'Date: 2026-01-19';
    RAISE NOTICE '====================================';
END $$;
