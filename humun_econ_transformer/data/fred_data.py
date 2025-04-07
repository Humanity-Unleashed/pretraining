import logging
import random
from typing import Dict, List, Tuple
from datasets import interleave_datasets, load_dataset, load_from_disk, Dataset
import ast
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

def convert_history_and_forecast_str(example: Dict):
    example['history'] = '<history>\n' + '\n'.join([f'({entry[0]}, {float(entry[1]):.1f})' for entry in example['history']] + [f'({entry[0]}, x)' for entry in example['forecast']]) + '\n</history>'
    example['forecast'] = '<forecast>\n' + '\n'.join([f'({entry[0]}, {float(entry[1]):.1f})' for entry in example['forecast']]) + '\n</forecast>'
    return example

def process_series(series_id, title, data_dict, max_prediction_window, context_multiplier, num_eval_chunks, max_cutoff_year=None):
    # Sort data by date
    sorted_data = sorted(data_dict.items(), key=lambda x: x[0])
    
    # Filter by cutoff year if specified
    if max_cutoff_year is not None:
        sorted_data = [item for item in sorted_data if 
                       datetime.strptime(item[0], '%Y-%m-%d').year >= max_cutoff_year]
    
    # Check if we have enough data after filtering
    if not sorted_data:
        return [], []
    
    chunks = []
    i = 0
    
    while i < len(sorted_data):
        # Determine random prediction window length for this chunk
        # Limited by max_prediction_window and remaining data
        max_possible = min(max_prediction_window, len(sorted_data) - i)
        if max_possible < 2:  # Need at least 2 points for prediction
            break
            
        pred_window = random.randint(2, max_possible)
        context_window = context_multiplier * pred_window
        
        # Check if we have enough data for the context window
        if i < context_window:
            # Not enough history, move to next position
            i += 1
            continue
            
        # Extract history and forecast portions
        history = sorted_data[i-context_window:i]
        forecast = sorted_data[i:i+pred_window]
        
        # Validate all values in the chunk
        if (len(history) == context_window and 
            len(forecast) == pred_window and 
            all(value is not None and value != '.' for _, value in history) and
            all(value is not None and value != '.' for _, value in forecast)):
            
            chunks.append((history, forecast))
        
        # Move to next position to start a new chunk
        i += pred_window
    
    # If we don't have enough chunks for both training and testing, return empty lists
    if len(chunks) <= num_eval_chunks:
        return [], []
    
    train_chunks = chunks[:-num_eval_chunks]
    test_chunks = chunks[-num_eval_chunks:]
    
    return train_chunks, test_chunks

def get_fred_data(
    raw_dataset_path: str,
    metadata_path: str,
    max_prediction_window: int,
    context_multiplier: int,
    num_eval_chunks: int,
    max_cutoff_year: int = None,
    seed=42,
    max_count=5000000,
    filters: Dict = {"frequency": "Monthly"},  # Default to Monthly
):
    """
    Loads FRED time series CSV file and metadata file and returns a HuggingFace Dataset for training and evaluation.
    Processes raw data into chunks with variable prediction windows and context sizes.
    Converts 'history' and 'forecast' columns from lists of tuples to strings.

    Args:
        raw_dataset_path: Path to raw dataset CSV file with time series data.
        metadata_path: Path to metadata CSV file.
        max_prediction_window: Maximum number of values to predict (actual window size will be randomly chosen between 2 and this value).
        context_multiplier: Multiplier to determine context window size (context_window = context_multiplier * prediction_window).
        num_eval_chunks: Number of chunks to use for evaluation from each time series.
        max_cutoff_year: Only consider data points from this year onwards (optional).
        seed: Random seed to set for dataset shuffling.
        max_count: Maximum number of samples to include in the dataset.
        filters: Dictionary of column:value pairs to filter metadata on.

    Returns:
        HuggingFace Dataset for train and test.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read metadata and apply filters
    metadata_df = pd.read_csv(metadata_path)

    for column, value in filters.items():
        if column in metadata_df.columns:
            # For string columns, do a case-insensitive comparison
            if metadata_df[column].dtype == "object":
                metadata_df = metadata_df[
                    metadata_df[column].str.lower() == str(value).lower()
                ]
            else:
                metadata_df = metadata_df[metadata_df[column] == value]
        else:
            raise ValueError(f"Erroneous filter provided: {column}")
    
    # Read and process the raw dataset
    df = pd.read_csv(raw_dataset_path)
    
    def safe_literal_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Skipping invalid entry: {x} - Error: {e}")
                return None  # Return original value if parsing fails
        return x

    df['data'] = df['data'].apply(safe_literal_eval)
    df = df.dropna(subset=['data'])

    train_data, test_data = [], []
    skipped_series = 0
    processed_series = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing time series"):
        train_chunks, test_chunks = process_series(
            row['series_id'], 
            row['title'], 
            row['data'], 
            max_prediction_window, 
            context_multiplier, 
            num_eval_chunks, 
            max_cutoff_year
        )
        
        if not train_chunks and not test_chunks:
            skipped_series += 1
            continue
            
        processed_series += 1
        
        for history, forecast in train_chunks:
            train_data.append({
                'series_id': row['series_id'], 
                'title': row['title'], 
                'history': history, 
                'forecast': forecast
            })
        
        for history, forecast in test_chunks:
            test_data.append({
                'series_id': row['series_id'], 
                'title': row['title'], 
                'history': history, 
                'forecast': forecast
            })
    
    # Log processing statistics
    logging.info(f"Processed {processed_series} time series, skipped {skipped_series} time series")
    logging.info(f"Created {len(train_data)} training examples and {len(test_data)} test examples")
    
    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    # Filter dataset by selected series ids from metadata
    selected_ids = metadata_df["id"].tolist()
    def matching_id_and_no_missing_values(example):
        # This is now redundant as we already filtered during processing, but keeping for safety
        return example["series_id"] in selected_ids

    # Log status about filtering
    total_train = len(train_dataset)
    total_test = len(test_dataset)
    
    filtered_train_dataset = train_dataset.filter(matching_id_and_no_missing_values)
    filtered_test_dataset = test_dataset.filter(matching_id_and_no_missing_values)
    
    logging.info(f"Filtered train dataset: {total_train} → {len(filtered_train_dataset)} examples")
    logging.info(f"Filtered test dataset: {total_test} → {len(filtered_test_dataset)} examples")

    # Convert list of time series values to newline-separated values, with forecast timestamps included in history
    filtered_train_dataset = filtered_train_dataset.map(convert_history_and_forecast_str)
    filtered_test_dataset = filtered_test_dataset.map(convert_history_and_forecast_str)

    # Limit dataset size if needed
    train_dataset = filtered_train_dataset.select(range(min(max_count, len(filtered_train_dataset))))
    test_dataset = filtered_test_dataset.select(range(min(max_count, len(filtered_test_dataset))))
    
    # Shuffle the datasets
    train_dataset = train_dataset.shuffle(seed=seed)
    test_dataset = test_dataset.shuffle(seed=seed)
    
    return train_dataset, test_dataset