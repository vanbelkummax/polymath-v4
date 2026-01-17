# Polymath v4 Makefile
# Run with: make <target>

.PHONY: help setup db-init ingest concepts assets github sync health test clean

# Default target
help:
	@echo "Polymath v4 - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup        - Install dependencies and create .env"
	@echo "  make db-init      - Initialize PostgreSQL schema"
	@echo ""
	@echo "Ingestion:"
	@echo "  make ingest PDF=path/to/paper.pdf  - Ingest a PDF"
	@echo "  make ingest-batch DIR=path/to/dir  - Ingest directory of PDFs"
	@echo "  make concepts     - Submit batch concept extraction"
	@echo "  make concepts-process  - Process completed concept results"
	@echo ""
	@echo "Discovery:"
	@echo "  make assets       - Discover GitHub/HF assets in papers"
	@echo "  make github URL=https://github.com/owner/repo  - Ingest GitHub repo"
	@echo "  make github-queue - Process GitHub queue"
	@echo ""
	@echo "Maintenance:"
	@echo "  make sync         - Sync to Neo4j"
	@echo "  make health       - Run health check"
	@echo "  make skills       - List skills pending promotion"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Remove cache files"

# Setup
setup:
	pip install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env - please edit it"; fi

db-init:
	@for f in schema/*.sql; do \
		echo "Running $$f..."; \
		psql -U polymath -d polymath -f "$$f"; \
	done
	@echo "Schema initialized"

# Ingestion
ingest:
ifndef PDF
	$(error PDF is required. Usage: make ingest PDF=path/to/paper.pdf)
endif
	python scripts/ingest_pdf.py "$(PDF)"

ingest-batch:
ifndef DIR
	$(error DIR is required. Usage: make ingest-batch DIR=path/to/dir)
endif
	python scripts/ingest_pdf.py "$(DIR)"/*.pdf --workers 4

concepts:
	python scripts/batch_concepts.py --submit --limit 1000
	@echo ""
	@echo "Check status with: make concepts-status"

concepts-status:
	python scripts/batch_concepts.py --status

concepts-process:
	python scripts/batch_concepts.py --process

# Discovery
assets:
	python scripts/discover_assets.py --github --add-to-queue

github:
ifndef URL
	$(error URL is required. Usage: make github URL=https://github.com/owner/repo)
endif
	python scripts/github_ingest.py "$(URL)"

github-queue:
	python scripts/github_ingest.py --queue --limit 10

# Maintenance
sync:
	python scripts/sync_neo4j.py --incremental

health:
	python scripts/system_report.py --quick

skills:
	python scripts/promote_skill.py --list

# Development
test:
	python -c "from lib.config import config; print('Config OK:', config.POSTGRES_DSN)"
	python -c "from lib.embeddings.bge_m3 import BGEEmbedder; e = BGEEmbedder(); print('Embedder OK:', e.embed_single('test').shape)"
	python -c "import psycopg2; c = psycopg2.connect('dbname=polymath user=polymath host=/var/run/postgresql'); print('Postgres OK:', c.status)"
	@echo "All tests passed"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cache cleaned"
