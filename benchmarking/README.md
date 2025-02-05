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
* stats per dataset, per model ? (can identify what types of datasets it excels at)
* use a config file for benchmarks (.toml) ? 
* capture tasks in notion
* redo arima 
* refactor CRPS to be calculated on multiple timeseries (currently just multiple forecasts of the same timeseries)
* produce first-run table of results
* docker file for creating a run of models + metrics (without relying on user-session being active)

Metrics: 
* test n_steps, more accurate at monthly, quarterly or yearly etc. 
* MAE distribution across all datasets (per model)
* token overhead - with 40+ years of monthly data, predicting 1 year, does the token overhead of providing 39+ years of data degrade performance? test vs. 10 years etc. (record n_tokens in logs)

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


