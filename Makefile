.PHONY: install test lint eval run clean help

help:
	@echo "Available commands:"
	@echo "  make install  - Install production and development dependencies"
	@echo "  make test     - Run full pytest test suite"
	@echo "  make lint     - Run Ruff linter"
	@echo "  make eval     - Execute evaluation harness on frozen test split"
	@echo "  make run      - Launch Streamlit triage assistant prototype UI"
	@echo "  make api      - Launch FastAPI REST serving microservice"

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	ruff check .

eval:
	python -m eval.run_eval --mode rules --split test

run:
	streamlit run app.py

api:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".hypothesis" -exec rm -rf {} +
