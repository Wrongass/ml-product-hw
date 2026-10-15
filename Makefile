.PHONY: setup check env s3-up s3-down ingest

setup: env
	uv sync
	uv run pre-commit install

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

env:
	test -f .env || cp .env.example .env

s3-up: env
	docker compose up -d

s3-down:
	docker compose down

ingest: env
	uv run python -m ml_product_hw.data.ingest
