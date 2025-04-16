# Economics Transformer
Code for finetuning Economics Transformer.

## TODOS
- Train different baseline models and pass to Aiden for evaluation
- Ablate over different models and sizes (eg. Qwen-2.5 Instruct, Llama Instruct models), as well as full finetuning vs LoRA (need to sweep over rank)
- Integrate other optimizers (eg. SOAP) into codebase

## Setup

- First install torch, then run `pip install -r requirements.txt`, then run `pip install -e .`.
- To run the training script, you need to first download the FRED time series data and the metadata. These are available on GCS at `gs://humun-storage/path/in/bucket/final_filtered_FRED_data.csv` and `gs://humun-storage/path/in/bucket/all_fred_metadata.csv`. To download these, refer to the first part of the [tutorial](https://humanity-unleashed.notion.site/FRED-Time-Series-Scraping-Tutorial-51774df4e0a5484e8458ae4665e53664) written by the Data Collection team to get the file containing the keys to access the GC space as well as a python example script to download both files. You will need to pass them as `raw_data_path` and `metadata_path` arguments to the SFT script (see more below).

### Dataset
The dataset at `gs://humun-storage/path/in/bucket/final_filtered_FRED_data.csv` gets preprocessed in the following ways (see the `get_fred_data` function in `humun_econ_transformer/data/fred_data.py`; a standalone script is also available at `humun_econ_transformer/data/preprocess_fred_data.py`):
- Takes each time series and splits into chunks; `max_prediction_window` specifies the max length of the forecast (where we select a random value from 2 to `max_prediction_window` for each chunk), and we provide `context_window = context_multiplier * prediction_window` additional values for the forecast.
- The final chunk (when sorted in chronological order) is passed to the test set. One can set `max_cutoff_year` to ensure the test set only contains samples from a specific year onwards. The remaining chunks are used for training. This yields a train and test DataFrame with columns `['series_id', 'title', 'history', 'forecast']` where the `history` column contains the first `context_window` values of the chunk and `forecast` column yields the remaining `prediction_window` values of the chunk, saved as list of tuples `('YYYY-MM-DD', v)`.

## Supervised Fine-tuning Code Instructions

The main training script can be found in `humun_econ_transformer/train_sft.py`. Once you have performed the setup instructions above, you can run this script with Deepspeed via the command `deepspeed --module humun_econ_transformer.train_sft`. The important arguments to pass when running this script are the following:
 * `--raw_data_path`: The path to the raw FRED `final_filtered_FRED_data.csv` file
 * `--metadata_path`: The path to the FRED metadata `metadata.csv` file, eg. `datasets/all_fred_metadata.csv`
 * `--processed_dataset_path`: An optional argument to set a folder path to save the processed SFT dataset so it will be retrieved if you run the same script again (processing the dataset takes awhile, so this will save time in future runs), eg. `datasets/processed_split`.
 * `--input_key`: The key used by the SFT trainer as the input prompt. This should be `history` for most purposes, but you can modify the default prompt, set this to a different field, and pass in a different `input_key` for training.
 * `--output_key`: The key used by the SFT trainer as the expected output, i.e. identifying the tokens that loss is computed on. This should be `forecast` for most purposes.


## Launch Script Instructions

To run locally, an example script is in `scripts/train_sft.sh`. If you're running code on a SLURM cluster, an example sbatch script which allows for multi-GPU training is given in `scripts/slurm_train_sft.sh`.

Pretraining project notion board: https://www.notion.so/humanity-unleashed/Pretraining-131d57b83b5181ebb282ff6569458c59




