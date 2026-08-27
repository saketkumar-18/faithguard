.PHONY: install test test-models serve build run logs clean benchmark train eval reports

PY ?= .venv/Scripts/python.exe
ifeq ($(OS),)
PY ?= .venv/bin/python
endif

install:            ## create venv + install deps (CPU torch)
	python -m venv .venv
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/cpu torch
	$(PY) -m pip install -r requirements.txt

test:               ## fast unit + API tests (no model downloads)
	$(PY) -m pytest -q

test-models:        ## integration tests with real NLI + embedding models
	$(PY) -m pytest -q -m models

benchmark:          ## build the SQuAD-derived benchmark dataset
	$(PY) scripts/build_benchmark.py

train:              ## train the hallucination classifier (~1h CPU first run, cached after)
	$(PY) scripts/train_classifier.py

eval:               ## end-to-end faithfulness eval: baseline vs guarded (needs LLM key)
	$(PY) scripts/evaluate_faithfulness.py

reports:            ## render markdown reports from JSON outputs
	$(PY) scripts/make_reports.py

serve:              ## run the API locally on :8000
	$(PY) -m uvicorn faithguard.api.app:app --host 0.0.0.0 --port 8000

build:              ## build the production Docker image
	docker build -t faithguard .

run:                ## run the production container (reads .env)
	docker compose up -d

logs:               ## tail container logs
	docker compose logs -f

clean:              ## remove caches
	rm -rf .pytest_cache faithguard/**/__pycache__
