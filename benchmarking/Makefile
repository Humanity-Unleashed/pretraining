ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
DATASET_DIR := data/fred

.PHONY: help dataset model

.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  dataset           Download datasets to the data/fred/ directory."
	@echo "  model repo=<name> Download a Hugging Face model to the local cache."

dataset:
	mkdir -p $(DATASET_DIR) && \
	wget -O $(DATASET_DIR)/unemployment_counties.csv https://huggingface.co/datasets/yatsbm/FRED/resolve/main/unemployment/unemployment_counties.csv?download=true || \
		{ echo "Error: Failed to download unemployment_counties.csv."; exit 1; } && \
	wget -O $(DATASET_DIR)/unemployment_states.csv https://huggingface.co/datasets/yatsbm/FRED/resolve/main/unemployment/unemployment_states.csv?download=true || \
		{ echo "Error: Failed to download unemployment_states.csv."; exit 1; }

model:
	@echo "Downloading model from Hugging Face repository: $(repo)"
	@if [ -z "$(repo)" ]; then \
		echo "Error: Please specify the 'repo' variable. Example: make model repo=EleutherAI/gpt-neo-2.7B"; \
		exit 1; \
	fi
	@huggingface-cli download $(repo) --include "*.safetensors" --repo-type model || \
		{ echo "Error: Failed to download model. Please check the repository name and your Hugging Face token."; exit 1; }
	@echo "Model downloaded successfully to Hugging Face cache."
