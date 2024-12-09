<div align="center">

A tool to systematically test LLMs with time-series forecasting and policy NLP tasks.  

**Humun Org**
</div>

## Description

Prompt method inspired by CiK forecasting ([paper](https://arxiv.org/abs/2410.18959))[github](https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py).

## Evaluation Metrics

* Continuous Ranked Probability Score (CRPS; Gneiting & Raftery (2007))
* Region of Interest CRPS (CiK; )

## Models

| Model Name | Code Repo | Pre-trained Repo | Paper |
| - | - | - | - |
| Llama-3.1-8B-Instruct | [llama-stack](https://github.com/meta-llama/llama-stack) | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)  | - | 
| Ministral-8B-Inst-2410 | [mistral-inference](https://github.com/mistralai/mistral-inference) | [HuggingFace](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) | - |
| Informer | [Informer2020](https://github.com/zhouhaoyi/Informer2020) | - | [arXiv](https://arxiv.org/abs/2012.07436) |  

## Time-series Data

| Name | Metadata | Link |
| - | - | - |
| FRED Unemployment | No | (https://huggingface.co/datasets/yatsbm/FRED) |


## Installation
> [!Note]
> UV is recommended to install project dependencies. Read Docs [here.](https://docs.astral.sh/uv/getting-started/features/#projects)

Using uv:
```bash
uv sync 
```

## Data download
> [!Note]
> Using the makefile download requires wget. 

```bash
make dataset
```

## Environment Variables 

**Create and implement?**
HUMUN_DATA_STORE, HUMUN_RESULT_CACHE  


