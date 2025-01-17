<div align="center">

A tool to systematically test LLMs with time-series forecasting and policy NLP tasks.  

**Humun Org**
</div>

## Description

Prompt method inspired by CiK forecasting ([paper](https://arxiv.org/abs/2410.18959))[github](https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py).


## Time-series Data

| Name | Metadata | Link |
| - | - | - |
| FRED Unemployment | No | (https://huggingface.co/datasets/yatsbm/FRED) |


## Installation
> [!Note]
> If using flash-attention-2 for attention mask configuration, see comments below to install correctly. 

Using uv:
```bash
# For usage see: https://docs.astral.sh/uv/getting-started/features/#projects
uv sync 
```

## Data download
> [!Note]
> Using the makefile download requires wget. 

```bash
make dataset
```

## Environment Variables 

requires `HF_CACHE`, `DATA_STORE`, `MODEL_STORE`


