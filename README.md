# Pretraining
Code for Pretraining Economics Transformer

Pretraining project notion board: https://www.notion.so/humanity-unleashed/Pretraining-131d57b83b5181ebb282ff6569458c59

## TO-DOs
* Set up direct prompt zero-shot baselines for FRED data (https://huggingface.co/datasets/yatsbm/FRED) and eventually FRED data collected by DataCollection
* Explore data generation strategies for (text, time series) pairs and evaluate on above baseline^

\[Rosie\] TO-DOS re: training codebase (01/02/25)
* Refactor code to use data from data collection team
* Try training with different models and sizes (using LoRA, etc)
* Incorporate other evaluation metrics (currently just next-token prediction loss)

\[Aiden\]  TO-DOs re: zero-shot benchmarks (02/02/25) 
* see [benchmarking readme.md](https://github.com/Humanity-Unleashed/pretraining/blob/main/benchmarking/README.md)
