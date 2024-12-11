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
> If using flash-attention-2 for attention mask configuration, see comments below to install correctly. 

Using uv:
```bash
# For usage see: https://docs.astral.sh/uv/getting-started/features/#projects
uv sync 
```

Flash-attention-2 is an algorithm used to increase efficiency in parallelisation. `torch` and `setup-tools` are required prior to installation as defined in `pyproject.toml`. Though for venv builds, flash attention needs to be installed using `uv pip install flash-attn --no-build-isolation`. This also requires `nvcc` to be available and the `CUDA_HOME` path to be defined. These are sometimes not defined on lightweight servers. 

To define these variables on the humun gpu server, run:
```
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
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


