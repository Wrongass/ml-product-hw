SEED ?= 42
VAL_SIZE ?= 5000
VERSION ?= v1

.PHONY: setup check env s3-up s3-down ingest data

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

data: env
	uv run python -m ml_product_hw.data.upload --version $(VERSION) --check-only
	uv run python -m ml_product_hw.data.download
	uv run python -m ml_product_hw.data.convert
	uv run python -m ml_product_hw.data.split --val-size $(VAL_SIZE) --seed $(SEED)
	uv run python -m ml_product_hw.data.stats
	uv run python -m ml_product_hw.data.upload --version $(VERSION)
