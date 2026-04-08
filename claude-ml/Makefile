# Urban Segmentation — developer convenience targets
.PHONY: install install-dev lint test train evaluate export serve docker-train docker-serve

PYTHON ?= python
DATA_ROOT ?= data
CHECKPOINT ?= outputs/best.ckpt

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebook]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
lint:
	ruff check src tests scripts
	mypy src/urban_seg

format:
	ruff format src tests scripts

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
test:
	pytest tests/ -v --cov=urban_seg --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --tb=short

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
prepare-data:
	$(PYTHON) scripts/prepare_data.py --root $(DATA_ROOT)

# ---------------------------------------------------------------------------
# ML lifecycle
# ---------------------------------------------------------------------------
train:
	$(PYTHON) scripts/train.py paths.data_root=$(DATA_ROOT)

train-dist:
	$(PYTHON) scripts/train.py +training=distributed paths.data_root=$(DATA_ROOT)

evaluate:
	$(PYTHON) scripts/evaluate.py checkpoint=$(CHECKPOINT)

export:
	$(PYTHON) scripts/export_model.py checkpoint=$(CHECKPOINT) export_dir=exports

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
serve:
	MODEL_PATH=exports/model.onnx uvicorn urban_seg.deployment.server:app \
		--host 0.0.0.0 --port 8080 --workers 1

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
docker-train:
	docker build -t urban-seg:train -f docker/Dockerfile.train .

docker-serve:
	docker build -t urban-seg:serve -f docker/Dockerfile.serve .

compose-up:
	docker compose -f docker/docker-compose.yml up -d

compose-down:
	docker compose -f docker/docker-compose.yml down

# ---------------------------------------------------------------------------
# Kubernetes
# ---------------------------------------------------------------------------
k8s-deploy:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/inference-deployment.yaml
	kubectl apply -f k8s/inference-service.yaml

k8s-train:
	kubectl apply -f k8s/training-job.yaml
