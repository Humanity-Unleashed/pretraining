.ONESHELL:
SHELL := bash

PYTHON := $(shell command -v python3 || echo "not_found")

.DEFAULT_GOAL := help

#* Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install           Create virtual environment, install dependencies." 
	@echo "  format            Run code formatters (black, isort)."

#* Install
.PHONY: install 
install:
	uv sync
	uv pip install flash-attn --no-build-isolation
	source .venv/bin/activate

#* Code Formatters
.PHONY: format
format:
	@echo "Running code formatters..."
	uv run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	uv run isort --settings-path pyproject.toml ./
	uv run black --config pyproject.toml ./
	@echo "Code formatting complete."

