.PHONY: help setup up down restart logs \
        etl train api dashboard \
        test lint format \
        db-shell db-reset clean

ENV_FILE := .env

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup"
	@echo "  setup         Copy .env.example to .env (if missing)"
	@echo ""
	@echo "Docker"
	@echo "  up            Build and start all services"
	@echo "  down          Stop all services"
	@echo "  restart       Restart all services"
	@echo "  logs          Tail logs for all services"
	@echo "  logs-api      Tail API logs"
	@echo "  logs-dash     Tail dashboard logs"
	@echo ""
	@echo "Pipeline"
	@echo "  etl           Run ETL pipeline (ingest → clean → transform → load)"
	@echo "  train         Train the ML model"
	@echo ""
	@echo "Dev"
	@echo "  api           Run API locally (outside Docker)"
	@echo "  dashboard     Run dashboard locally (outside Docker)"
	@echo "  test          Run test suite"
	@echo "  lint          Run ruff linter"
	@echo "  format        Run ruff formatter"
	@echo ""
	@echo "Database"
	@echo "  db-shell      Open psql shell inside the postgres container"
	@echo "  db-reset      Drop and reinitialize the database"
	@echo ""
	@echo "  clean         Remove __pycache__ and .pyc files"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup:
	@if [ ! -f $(ENV_FILE) ]; then \
		cp .env.example $(ENV_FILE); \
		echo "Created $(ENV_FILE) — fill in your credentials."; \
	else \
		echo "$(ENV_FILE) already exists, skipping."; \
	fi

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

up: setup
	docker compose --env-file $(ENV_FILE) up --build -d

down:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-dash:
	docker compose logs -f dashboard

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

etl:
	docker compose --env-file $(ENV_FILE) run --rm etl python -m services.etl.ingest
	docker compose --env-file $(ENV_FILE) run --rm etl python -m services.etl.clean
	docker compose --env-file $(ENV_FILE) run --rm etl python -m services.etl.transform
	docker compose --env-file $(ENV_FILE) run --rm etl python -m services.etl.load

train:
	docker compose --env-file $(ENV_FILE) run --rm ml python -m services.ml.train

# ---------------------------------------------------------------------------
# Local dev (no Docker)
# ---------------------------------------------------------------------------

api:
	uvicorn services.api.main:app --reload --port $${API_PORT:-8000}

dashboard:
	streamlit run services/dashboard/app.py --server.port $${DASHBOARD_PORT:-8501}

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test:
	pytest tests/ -v

lint:
	ruff check services/

format:
	ruff format services/

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

db-shell:
	docker compose exec postgres psql -U $${POSTGRES_USER:-oracle} -d $${POSTGRES_DB:-urban_demand}

db-reset:
	docker compose down -v
	docker compose --env-file $(ENV_FILE) up -d postgres
	@echo "Database reset — run 'make etl' to repopulate."

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
