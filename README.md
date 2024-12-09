# Pretraining
Code for Pretraining Economics Transformer

Pretraining project notion board: https://www.notion.so/humanity-unleashed/Pretraining-131d57b83b5181ebb282ff6569458c59

## TO-DOs (Updated 12/06)
* Set up direct prompt zero-shot baselines for FRED data (https://huggingface.co/datasets/yatsbm/FRED) and eventually FRED data collected by DataCollection
* Explore data generation strategies for (text, time series) pairs and evaluate on above baseline^

  
\[Aiden\] Remaining Tasks (12/09) 
* unsure of project structure (benchmarking package + finetuning) - to fix/clarify
* set up to use server datasets repo
* get Make added to server to utilize makefiles?  
* set up n-retries in `clients/huggingface.py` to send multiple prompts to create a forecast distribution.
* use `utils/parse.py` to parse string output to compute metrics like CRPS, MAE etc. on. 
* use `llm-format-enforcer` to enforce output tokens allowed by a model (stricter, less hallucinations?)


