SEED ?= 42
VAL_SIZE ?= 5000
VERSION ?= v1
IMAGE = ml-product-hw-train
GIT_COMMIT = $(shell git rev-parse HEAD)

.PHONY: setup check env s3-up s3-down ingest data infra-up infra-down train experiments export serve-up serve-down verify load-test

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

infra-up: env
	docker build -t $(IMAGE) .
	docker compose up -d --build --wait

infra-down:
	docker compose down

train: CONFIG ?= configs/simple_cnn.yaml
train: env
	docker run --rm --network ml-product-hw --env-file .env \
		-e S3_ENDPOINT_URL=http://rustfs:9000 \
		-e MLFLOW_TRACKING_URI=http://mlflow:5000 \
		-e GIT_COMMIT=$(GIT_COMMIT) \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/configs:/app/configs \
		$(IMAGE) python -m ml_product_hw.train.train --config $(CONFIG)

experiments: CONFIG ?= configs/grid.yaml
experiments: env
	GIT_COMMIT=$(GIT_COMMIT) uv run python -m ml_product_hw.train.grid --config $(CONFIG) --image $(IMAGE)

export: env
	uv run python -m ml_product_hw.serving.export

serve-up: env
	docker compose --profile serving up -d --build --wait

serve-down:
	docker compose --profile serving down

verify: env
	uv run python -m ml_product_hw.serving.verify

load-test: env
	uv run python -m ml_product_hw.serving.load_test
