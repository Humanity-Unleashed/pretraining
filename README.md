# Pretraining
Code for Pretraining Economics Transformer

Pretraining project notion board: https://www.notion.so/humanity-unleashed/Pretraining-131d57b83b5181ebb282ff6569458c59

## TO-DOs (Updated 12/06)
* Set up direct prompt zero-shot baselines for FRED data (https://huggingface.co/datasets/yatsbm/FRED) and eventually FRED data collected by DataCollection
* Explore data generation strategies for (text, time series) pairs and evaluate on above baseline^

\[Rosie\] TO-DOS re: training codebase (01/02/25)
* Refactor code to use data from data collection team
* Try training with different models and sizes (using LoRA, etc)
* Incorporate other evaluation metrics (currently just next-token prediction loss)
  
\[Aiden\]  TO-DOs re: zero-shot benchmarks (12/23) 
* set up to use server datasets repo
* implement multiple models (instruct LLMs and general timeseries e.g. ARIMA)
* create metrics functions (MAE, CRPS etc.)
* unit tests
* create a makefile (?) for systematic inference generation / benchmarking using a config

