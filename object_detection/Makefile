.PHONY: install lint test train eval export serve docker-train docker-serve k8s-train clean

PYTHON ?= python
DATA_ROOT ?= /data/coco
CHECKPOINT ?= outputs/best.ckpt

# ── Development ─────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/
	mypy src/

format:
	ruff format src/ tests/ scripts/

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -x -q

# ── Data ────────────────────────────────────────────────────────────────────

data-download:
	$(PYTHON) scripts/prepare_data.py download-coco --root $(DATA_ROOT)

data-verify:
	$(PYTHON) scripts/prepare_data.py verify-coco --root $(DATA_ROOT)

data-stats:
	$(PYTHON) scripts/prepare_data.py stats --root $(DATA_ROOT)

# ── Training ────────────────────────────────────────────────────────────────

train:
	PYTHONPATH=src $(PYTHON) scripts/train.py

train-distributed:
	PYTHONPATH=src $(PYTHON) scripts/train.py training=distributed

train-dino:
	PYTHONPATH=src $(PYTHON) scripts/train.py model=dino_r50

# ── Evaluation ──────────────────────────────────────────────────────────────

eval:
	PYTHONPATH=src $(PYTHON) scripts/evaluate.py \
		--checkpoint $(CHECKPOINT) \
		--data-root $(DATA_ROOT) \
		--output-dir outputs/eval

# ── Export ──────────────────────────────────────────────────────────────────

export-onnx:
	PYTHONPATH=src $(PYTHON) scripts/export_model.py checkpoint=$(CHECKPOINT)

export-trt:
	PYTHONPATH=src $(PYTHON) scripts/export_model.py \
		checkpoint=$(CHECKPOINT) deployment=tensorrt

# ── Serving ─────────────────────────────────────────────────────────────────

serve:
	PYTHONPATH=src $(PYTHON) -c \
		"from urban_det.deployment.server import serve; serve('outputs/model.onnx')"

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Docker ───────────────────────────────────────────────────────────────────

docker-build-train:
	docker build -f docker/Dockerfile.train -t urban-det-train:latest .

docker-build-serve:
	docker build -f docker/Dockerfile.serve -t urban-det-serve:latest .

docker-build-edge:
	docker build -f docker/Dockerfile.edge -t urban-det-edge:latest .

# ── Kubernetes ───────────────────────────────────────────────────────────────

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/

k8s-train:
	kubectl apply -f k8s/namespace.yaml -f k8s/training-job.yaml

k8s-status:
	kubectl get all -n urban-det

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info
