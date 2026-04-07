# Adaptive Deep Networks Makefile
.PHONY: help install test lint format clean quick full paper-metrics

# Default target
help:
	@echo "Adaptive Deep Networks - Available Targets:"
	@echo ""
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make lint          - Run linting (black, ruff, mypy)"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean generated files"
	@echo "  make quick         - Quick validation (quick mode experiments)"
	@echo "  make full          - Full experiment suite"
	@echo "  make paper-metrics - Generate paper metrics"
	@echo "  make core          - Run core experiments"
	@echo "  make validate      - Run validation experiments"
	@echo "  make train-paper-small  - Train small model + strict paper alignment check"
	@echo "  make train-paper-t4     - Train T4 model + paper preset (T4 VRAM caps) + alignment check"
	@echo "  make train-paper-medium - Train medium model + strict paper alignment check"
	@echo "  make train-paper-large  - Train large model + strict paper alignment check"
	@echo ""

# Installation
install:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Linting and formatting
lint:
	black --check src/ experiments/ scripts/ tests/
	ruff check src/ experiments/ scripts/ tests/
	mypy src/

format:
	black src/ experiments/ scripts/ tests/

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/

# Experiment targets
quick:
	python experiments/run_experiments_unified.py --all --quick

full:
	python experiments/run_experiments_unified.py --all

core:
	python experiments/run_experiments_unified.py --category core

validate:
	python experiments/run_experiments_unified.py --category validation

paper-metrics:
	python experiments/run_experiments_unified.py --category paper

# List experiments
list:
	python experiments/run_experiments_unified.py --list

# Paper-aligned training wrappers (requires OUTPUT_DIR)
train-paper-small:
	@if [ -z "$(OUTPUT_DIR)" ]; then echo "Usage: make train-paper-small OUTPUT_DIR=results/small_paper"; exit 1; fi
	bash scripts/training/run_with_paper_alignment_check.sh --model-size small --output-dir "$(OUTPUT_DIR)"

train-paper-t4:
	@if [ -z "$(OUTPUT_DIR)" ]; then echo "Usage: make train-paper-t4 OUTPUT_DIR=results/t4_paper"; exit 1; fi
	bash scripts/training/run_with_paper_alignment_check.sh --model-size t4 --output-dir "$(OUTPUT_DIR)"

train-paper-medium:
	@if [ -z "$(OUTPUT_DIR)" ]; then echo "Usage: make train-paper-medium OUTPUT_DIR=results/medium_paper"; exit 1; fi
	bash scripts/training/run_with_paper_alignment_check.sh --model-size medium --output-dir "$(OUTPUT_DIR)"

train-paper-large:
	@if [ -z "$(OUTPUT_DIR)" ]; then echo "Usage: make train-paper-large OUTPUT_DIR=results/large_paper"; exit 1; fi
	bash scripts/training/run_with_paper_alignment_check.sh --model-size large --output-dir "$(OUTPUT_DIR)"
