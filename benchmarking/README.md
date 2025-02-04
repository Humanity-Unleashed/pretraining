<div align="center">

A tool to systematically test LLMs with time-series forecasting and policy NLP tasks.  

**Humun Org**
</div>

## Description

Instruct prompt method inspired by CiK forecasting [ [paper](https://arxiv.org/abs/2410.18959) | [github](https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py) ].


## TODOs:

* use n_steps instead of forecast_split,
* log result metrics, hardware setup + usage stats (pyviz/pyspy etc.), time elapsed, number of datasets,
* preflight checks (hardware setup + env vars etc.) - log, warn and raise errors
* outputs to JSON not pkl
* stats per dataset, per model ? (can identify what types of datasets it excels at)
* use a config file for benchmarks (.toml) ? 
* capture tasks in notion
* redo arima 
* truncation warning on inference 
* refactor CRPS to be calculated on multiple timeseries (currently just multiple forecasts of the same timeseries)
* produce first-run table of results
* docker file for creating a run of models + metrics (without relying on user-session being active)

## Time-series Data

| Name | Metadata | Link |
| - | - | - |
| FRED Unemployment | No | (https://huggingface.co/datasets/yatsbm/FRED) |


## Installation
> [!Note]
> If using flash-attention-2 for attention mask configuration, see comments below to install correctly. 

```bash
make install
```

## Data download
> [!Note]
> Using the makefile download requires wget. 

```bash
make dataset
```

## Environment Variables 

requires `HF_HOME`, `HF_TOKEN_PATH`, `DATA_STORE`, `MODEL_STORE`

In order to keep auth tokens in user directories.
`HF_STORED_TOKENS_PATH` follows `HF_TOKEN_PATH`
 see: https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L150


