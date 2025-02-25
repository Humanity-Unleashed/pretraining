import logging
from typing import Dict, List
from datasets import interleave_datasets, load_dataset, load_from_disk

import pandas as pd


def convert_history_and_forecast_str(example: Dict):
    example['history'] = '<history>\n' + '\n'.join([f'({entry[0]}, {float(entry[1]):.1f})' for entry in example['history']] + [f'({entry[0]}, x)' for entry in example['forecast']]) + '\n</history>'
    example['forecast'] = '<forecast>\n' + '\n'.join([f'({entry[0]}, {float(entry[1]):.1f})' for entry in example['forecast']]) + '\n</forecast>'
    return example

def get_fred_data(
    train_dataset_path: str,
    test_dataset_path: str,
    metadata_path: str,
    seed=42,
    max_count=5000000,
    filters: Dict = {"frequency": "Monthly"},  # Default to Monthly
):
    """
    Loads FRED time series parquet file and metadata file and returns a HuggingFace Dataset for training and evaluation.
    Converts 'history' and 'forecast' columns from lists of tuples to strings.

    Args:
        train_dataset_path: Path to train dataset parquet file with time series data.
        test_dataset_path: Path to test dataset parquet file with time series data.
        metadata_path: Path to metadata CSV file.
        seed: Random seed to set for dataset shuffling.
        max_count: Maximum number of samples to include in the dataset.
        test_size: Fraction of test split of dataset. Only used if `return_eval` is True.
        return_eval: Return test split of dataset.
        filters: Dictionary of column:value pairs to filter metadata on.

    Returns:
        HuggingFace Dataset for train and test (if `return_eval` is set to True).
    """
    # Read metadata and apply filters.
    metadata_df = pd.read_csv(metadata_path)

    for column, value in filters.items():
        if column in metadata_df.columns:
            # For string columns, do a case-insensitive comparison.
            if metadata_df[column].dtype == "object":
                metadata_df = metadata_df[
                    metadata_df[column].str.lower() == str(value).lower()
                ]
            else:
                metadata_df = metadata_df[metadata_df[column] == value]
        else:
            raise ValueError(f"Erroneous filter provided: {column}")
    

    # Load time series dataset in HuggingFace.
    train_dataset = load_dataset("parquet", data_files=train_dataset_path)['train']
    test_dataset = load_dataset("parquet", data_files=test_dataset_path)['train']

    # Filter dataset by missing values and match series id to those selected
    selected_ids = metadata_df["id"].tolist()
    def matching_id_and_no_missing_values(example):
        #@TODO: Currently filtering out any time series with any missing values in history and forecast
        return example["series_id"] in selected_ids and not any(value == '.' for _, value in example['history']) and not any(value == '.' for _, value in example['forecast'])

    filtered_train_dataset = train_dataset.filter(matching_id_and_no_missing_values)
    filtered_test_dataset = test_dataset.filter(matching_id_and_no_missing_values)

    # Convert list of time series values to newline-separated values, with forecast timestamps included in history.
    filtered_train_dataset = filtered_train_dataset.map(convert_history_and_forecast_str)
    filtered_test_dataset = filtered_test_dataset.map(convert_history_and_forecast_str)

    train_dataset = filtered_train_dataset.select(range(min(max_count, len(filtered_train_dataset))))
    test_dataset = filtered_test_dataset.select(range(min(max_count, len(filtered_test_dataset))))
    return train_dataset, test_dataset