# Polymath v4 Quickstart

> **Get from zero to running in 10 minutes.**

---

## Prerequisites

```bash
# Required
- Python 3.11+
- PostgreSQL 15+ with pgvector
- Docker (for Neo4j)
- NVIDIA GPU (for embeddings)
- Google Cloud service account (for Gemini)
```

---

## Step 1: Environment Setup (2 min)

```bash
cd /home/user/polymath-v4

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit environment
cp .env.example .env
vim .env  # Set your credentials
```

**.env contents:**
```bash
POSTGRES_DSN=dbname=polymath user=polymath host=/var/run/postgresql
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GITHUB_TOKEN=ghp_your_token_here  # Optional but recommended
```

---

## Step 2: Database Setup (2 min)

```bash
# Create database
sudo -u postgres createdb polymath
sudo -u postgres createuser polymath

# Run migrations
psql -U polymath -d polymath -f schema/001_core.sql
psql -U polymath -d polymath -f schema/002_concepts.sql
psql -U polymath -d polymath -f schema/003_code.sql
psql -U polymath -d polymath -f schema/004_skills.sql

# Start Neo4j
docker run -d \
  --name polymath-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  -v neo4j_data:/data \
  neo4j:5.15-community

# Apply Neo4j constraints
cat schema/005_neo4j.cypher | docker exec -i polymath-neo4j cypher-shell -u neo4j -p your_neo4j_password
```

---

## Step 3: Ingest Your First Paper (3 min)

```bash
# Single paper (includes embeddings)
python scripts/ingest_pdf.py /path/to/paper.pdf

# Batch of papers
python scripts/ingest_pdf.py /path/to/papers/*.pdf --workers 4

# Check result
psql -U polymath -d polymath -c "SELECT title, (SELECT COUNT(*) FROM passages p WHERE p.doc_id = d.doc_id) as passages FROM documents d ORDER BY created_at DESC LIMIT 5"
```

**Expected output:**
```
                    title                     | passages
----------------------------------------------+----------
 Squidpy: a scalable framework for spatial... |       87
```

---

## Step 4: Extract Concepts (2 min)

```bash
# Submit batch job (50% cheaper)
python scripts/batch_concepts.py --submit --limit 1000

# Check status (runs async)
python scripts/batch_concepts.py --status

# When complete, process results
python scripts/batch_concepts.py --process

# Verify
psql -U polymath -d polymath -c "SELECT concept_type, COUNT(*) FROM passage_concepts GROUP BY concept_type ORDER BY COUNT(*) DESC"
```

**Expected output:**
```
 concept_type | count
--------------+-------
 method       |   342
 entity       |   289
 domain       |   156
 dataset      |    87
 problem      |    65
```

---

## Step 5: Discover & Ingest Code (2 min)

```bash
# Find GitHub repos mentioned in papers
python scripts/discover_assets.py --github --add-to-queue

# See what's in the queue
python scripts/github_ingest.py --list

# Ingest top priority repos
python scripts/github_ingest.py --queue --limit 5

# Or ingest a specific repo
python scripts/github_ingest.py https://github.com/scverse/squidpy
```

---

## Step 6: Sync to Neo4j (1 min)

```bash
# Incremental sync (recent changes only)
python scripts/sync_neo4j.py --incremental

# Verify
docker exec polymath-neo4j cypher-shell -u neo4j -p your_neo4j_password \
  "MATCH (n) RETURN labels(n)[0] as type, count(n) ORDER BY count(n) DESC LIMIT 5"
```

---

## Step 7: Search

```bash
# CLI search
python -c "
from lib.search.hybrid_search import HybridSearcher
s = HybridSearcher()
for r in s.search('spatial gene expression from histology', n=5):
    print(f'{r[\"score\"]:.3f} | {r[\"title\"][:60]}')
"
```

---

## Daily Operations

### Morning Routine
```bash
# Quick health check
python scripts/system_report.py --quick

# Process any pending batch jobs
python scripts/batch_concepts.py --process

# Ingest queued repos
python scripts/github_ingest.py --queue --limit 5
```

### Adding New Papers
```bash
# Drop PDFs in staging folder
cp new_papers/*.pdf /home/user/work/staging/

# Ingest
python scripts/ingest_pdf.py /home/user/work/staging/*.pdf

# Extract concepts
python scripts/batch_concepts.py --submit
```

### Weekly Maintenance
```bash
# Full system report
python scripts/system_report.py --full -o SYSTEM_STATUS.md

# Discover new assets
python scripts/discover_assets.py --recommend

# Review skill drafts
python scripts/promote_skill.py --check-all

# Full Neo4j sync
python scripts/sync_neo4j.py --full
```

---

## Verification Queries

### PostgreSQL
```sql
-- Document stats
SELECT
  COUNT(*) as docs,
  SUM((SELECT COUNT(*) FROM passages p WHERE p.doc_id = d.doc_id)) as passages,
  SUM((SELECT COUNT(*) FROM passage_concepts pc
       JOIN passages p ON pc.passage_id = p.passage_id
       WHERE p.doc_id = d.doc_id)) as concepts
FROM documents d;

-- Embedding coverage
SELECT
  COUNT(*) as total,
  COUNT(embedding) as embedded,
  ROUND(100.0 * COUNT(embedding) / COUNT(*), 1) as pct
FROM passages;

-- Queue status
SELECT status, COUNT(*) FROM repo_queue GROUP BY status;
```

### Neo4j
```cypher
// Node counts
MATCH (n) RETURN labels(n)[0] as type, count(n) ORDER BY count(n) DESC

// Cross-domain concepts
MATCH (c:Concept)<-[:MENTIONS]-(p:Passage)
WITH c, count(DISTINCT p) as mentions
WHERE mentions > 5
RETURN c.name, mentions ORDER BY mentions DESC LIMIT 20
```

---

## Troubleshooting

### "Connection refused" on Neo4j
```bash
docker restart polymath-neo4j
sleep 30
docker logs polymath-neo4j --tail 20
```

### Embeddings slow
```bash
# Check GPU
nvidia-smi

# Reduce batch size in lib/embeddings/bge_m3.py
BATCH_SIZE = 16  # Default is 32
```

### Batch API stuck
```bash
# Check GCP console or:
python scripts/batch_concepts.py --status --verbose
```

---

## Cost Tracking

After running, check costs:

```bash
# Estimate from passage count
psql -U polymath -d polymath -c "
SELECT
  COUNT(*) as passages,
  ROUND(COUNT(*) * 0.0001, 2) as concept_cost_usd,
  ROUND(COUNT(*) * 0.001 / 50, 2) as skill_cost_usd
FROM passages
WHERE embedding IS NOT NULL
"
```

---

## Next Steps

1. **Read** `ARCHITECTURE.md` for full system understanding
2. **Configure** skills in `skills/` directory
3. **Set up** automated ingestion with cron
4. **Integrate** with your research workflow

---

*v4.0 - Lean and execution-ready*
