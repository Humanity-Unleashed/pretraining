.ONESHELL:
SHELL := bash

PYTHON := $(shell command -v python3 || echo "not_found")

.DEFAULT_GOAL := help

#* Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install           Set up the Python virtual environment and install dependencies."
	@echo "  dataset           Download datasets to the data/fred/ directory."
	@echo "  model repo=<name> Download a Hugging Face model to the local cache."
	@echo "  format            Run code formatters (black, isort)."

#* Installation
.PHONY: install
install:
	@if [ "$(PYTHON)" = "not_found" ]; then \
		echo "Error: Python not found. Please ensure Python is installed and accessible."; \
		exit 1; \
	fi
	@echo "Using Python: $(PYTHON)"
	rm -rf .venv
	uv sync
	source .venv/bin/activate
	@echo "Environment setup complete."

#* Dataset Management
.PHONY: dataset
dataset:
	mkdir -p data/fred
	wget -O data/fred/unemployment_counties.csv https://huggingface.co/datasets/yatsbm/FRED/resolve/main/unemployment/unemployment_counties.csv?download=true || \
		{ echo "Error: Failed to download unemployment_counties.csv."; exit 1; }
	wget -O data/fred/unemployment_states.csv https://huggingface.co/datasets/yatsbm/FRED/resolve/main/unemployment/unemployment_states.csv?download=true || \
		{ echo "Error: Failed to download unemployment_states.csv."; exit 1; }
	@echo "Datasets downloaded successfully."

#* Model Management
.PHONY: model
model:
	@echo "Downloading model from Hugging Face repository: $(repo)"
	@if [ -z "$(repo)" ]; then \
		echo "Error: Please specify the 'repo' variable. Example: make model repo=EleutherAI/gpt-neo-2.7B"; \
		exit 1; \
	fi
	huggingface-cli download $(repo) --include "*.safetensors" --repo-type model || \
		{ echo "Error: Failed to download model. Please check the repository name and your Hugging Face token."; exit 1; }
	@echo "Model downloaded successfully to Hugging Face cache."

#* Code Formatters
.PHONY: format
format:
	@echo "Running code formatters..."
	uv run pyupgrade --exit-zero-even-if-changed --py38-plus **/*.py
	uv run isort --settings-path pyproject.toml ./
	uv run black --config pyproject.toml ./
	@echo "Code formatting complete."
