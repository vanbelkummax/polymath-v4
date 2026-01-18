#!/bin/bash
# Polymath v4 Status Dashboard
# Usage: bash scripts/status.sh

echo "=============================================="
echo "       POLYMATH v4 STATUS DASHBOARD"
echo "=============================================="
echo ""

# Database counts
echo "📊 DATABASE STATE"
echo "----------------------------------------------"
psql -U polymath -d polymath -t -c "
SELECT
    'Papers:        ' || COUNT(*) FROM documents
UNION ALL SELECT
    'Paper passages:' || COUNT(*) FROM passages
UNION ALL SELECT
    'Repositories:  ' || COUNT(*) FROM repositories
UNION ALL SELECT
    'Repo passages: ' || COUNT(*) FROM repo_passages
UNION ALL SELECT
    'Paper-repo links:' || COUNT(*) FROM paper_repo_links;
"
echo ""

# Running jobs
echo "🔄 RUNNING JOBS"
echo "----------------------------------------------"
REPO_PID=$(cat /home/user/logs/repo_ingest.pid 2>/dev/null)
if [ -n "$REPO_PID" ] && ps -p $REPO_PID > /dev/null 2>&1; then
    echo "✅ Repo ingestion running (PID: $REPO_PID)"
    # Get progress from log
    PROGRESS=$(grep -c "Processing:" /home/user/logs/repo_ingest_full.log 2>/dev/null || echo "0")
    echo "   Progress: $PROGRESS / 2059 repos"
else
    echo "⏹️  No repo ingestion running"
fi
echo ""

# Recent log entries
echo "📝 RECENT LOG (last 5 lines)"
echo "----------------------------------------------"
tail -5 /home/user/logs/repo_ingest_full.log 2>/dev/null || echo "No log file"
echo ""

echo "=============================================="
echo "Full log: tail -f /home/user/logs/repo_ingest_full.log"
echo "=============================================="
