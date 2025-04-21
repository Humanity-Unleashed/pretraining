import logging
import random
from typing import Dict, List, Tuple
from datasets import interleave_datasets, load_dataset, load_from_disk, Dataset
import ast
import os
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

def sample_power_law_window(max_possible, min_possible=2, alpha=2):
    if max_possible == min_possible: return min_possible

    values = list(range(min_possible, max_possible + 1))
    
    # Compute unnormalized probabilities then normalize
    probs = np.array([x ** alpha for x in values], dtype=np.float64)
    probs /= probs.sum()
    
    # Sample from distribution
    pred_window = random.choices(values, weights=probs, k=1)[0]
    return pred_window

def convert_history_and_forecast_str(example: Dict):
    example['history'] = '<history>\n' + '\n'.join([f'({entry[0]}, {float(entry[1]):.1f})' for entry in example['history']] + [f'({entry[0]}, x)' for entry in example['forecast']]) + '\n</history>'
    example['forecast'] = '<forecast>\n' + '\n'.join([f'({entry[0]}, {float(entry[1]):.1f})' for entry in example['forecast']]) + '\n</forecast>'
    return example

def process_series(series_id, title, data_dict, max_prediction_window, context_multiplier, num_eval_chunks, max_cutoff_year=None):
    # Sort data by date
    sorted_data = sorted(data_dict.items(), key=lambda x: x[0])
    
    # Check if we have enough data
    if not sorted_data:
        return [], []
    
    train_chunks = []
    test_chunks = []
    
    if max_cutoff_year is not None:
        # Split data into pre-cutoff and post-cutoff based on dates
        pre_cutoff_data = []
        post_cutoff_data = []
        
        for date, value in sorted_data:
            year = datetime.strptime(date, '%Y-%m-%d').year
            if year < max_cutoff_year:
                pre_cutoff_data.append((date, value))
            else:
                post_cutoff_data.append((date, value))
        
        # Generate train chunks from pre-cutoff data only
        i = 0
        while i < len(pre_cutoff_data):
            max_possible = min(max_prediction_window, len(pre_cutoff_data) - i)
            if max_possible < 2:
                break
                
            pred_window = sample_power_law_window(max_possible, 2)
            context_window = context_multiplier * pred_window
            
            if i < context_window:
                i += 1
                continue
                
            history = pre_cutoff_data[i-context_window:i]
            forecast = pre_cutoff_data[i:i+pred_window]
            
            if (len(history) == context_window and 
                len(forecast) == pred_window and 
                all(value is not None and value != '.' for _, value in history) and
                all(value is not None and value != '.' for _, value in forecast)):
                
                train_chunks.append((history, forecast))
            
            i += pred_window
        
        # Generate test chunks where forecast starts from max_cutoff_year
        # but history can include pre-cutoff data
        if post_cutoff_data:
            # We need to find points where we can start forecasting from the cutoff year
            # The history can include data from before the cutoff
            for test_start_idx in range(min(num_eval_chunks, len(post_cutoff_data))):
                # For each potential starting point in post-cutoff data
                if test_start_idx + 2 > len(post_cutoff_data):
                    # Need at least 2 points for forecasting
                    break
                    
                max_possible = min(max_prediction_window, len(post_cutoff_data) - test_start_idx)
                pred_window = sample_power_law_window(max_possible, 2)
                context_window = context_multiplier * pred_window
                
                # The forecast will be from post_cutoff_data
                forecast = post_cutoff_data[test_start_idx:test_start_idx+pred_window]
                
                # History can include pre-cutoff data
                # Find where this forecast starts in the overall timeline
                full_idx = len(pre_cutoff_data) + test_start_idx
                
                # If we don't have enough history, just use all available
                history_start_idx = max(0, full_idx - context_window)
                
                if history_start_idx < len(pre_cutoff_data):
                    # History spans both pre and post cutoff
                    pre_history_count = len(pre_cutoff_data) - history_start_idx
                    history = pre_cutoff_data[history_start_idx:] + post_cutoff_data[:test_start_idx]
                else:
                    # History is entirely from post cutoff
                    history_start_in_post = history_start_idx - len(pre_cutoff_data)
                    history = post_cutoff_data[history_start_in_post:test_start_idx]
                
                # Validate the chunk
                if (len(history) == context_window and 
                    len(forecast) == pred_window and 
                    all(value is not None and value != '.' for _, value in history) and
                    all(value is not None and value != '.' for _, value in forecast)):
                    
                    test_chunks.append((history, forecast))
    else:
        # If no cutoff year specified, create chunks from all data
        # and use the last num_eval_chunks for testing
        all_chunks = []
        i = 0
        
        while i < len(sorted_data):
            max_possible = min(max_prediction_window, len(sorted_data) - i)
            if max_possible < 2:
                break
                
            pred_window = sample_power_law_window(max_possible, 2)
            context_window = context_multiplier * pred_window
            
            if i < context_window:
                i += 1
                continue
                
            history = sorted_data[i-context_window:i]
            forecast = sorted_data[i:i+pred_window]
            
            if (len(history) == context_window and len(forecast) == pred_window):
                
                all_chunks.append((history, forecast))
            
            i += pred_window
        
        if len(all_chunks) <= num_eval_chunks:
            return [], []
            
        train_chunks = all_chunks[:-num_eval_chunks]
        test_chunks = all_chunks[-num_eval_chunks:]
    
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
        max_cutoff_year: Only consider forecast data points from this year onwards for test set (optional).
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

    # Filter dataset by selected series ids from metadata
    selected_ids = metadata_df["id"].tolist()
    df = df[df["series_id"].isin(selected_ids)]

    # Merge additional columns from metadata_df into df
    df = df.merge(
        metadata_df[["id", "frequency", "units"]],
        how="left",
        left_on="series_id",
        right_on="id"
    )
    df = df.drop(columns=["id"])

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
                'frequency': row['frequency'],
                'units': row['units'],
                'context_window': len(history),
                'forecast_window': len(forecast),
                'forecast_date_start': forecast[0][0],
                'forecast_date_end': forecast[-1][0],
                'history': history, 
                'forecast': forecast
            })
        
        for history, forecast in test_chunks:
            test_data.append({
                'series_id': row['series_id'], 
                'title': row['title'], 
                'frequency': row['frequency'],
                'units': row['units'],
                'context_window': len(history),
                'forecast_window': len(forecast),
                'forecast_date_start': forecast[0][0],
                'forecast_date_end': forecast[0][-1],
                'history': history, 
                'forecast': forecast
            })
    
    # Log processing statistics
    logging.info(f"Processed {processed_series} time series, skipped {skipped_series} time series")
    logging.info(f"Created {len(train_data)} training examples and {len(test_data)} test examples")
    
    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    # Convert list of time series values to newline-separated values, with forecast timestamps included in history
    filtered_train_dataset = train_dataset.map(convert_history_and_forecast_str)
    filtered_test_dataset = test_dataset.map(convert_history_and_forecast_str)

    # Limit dataset size if needed
    train_dataset = filtered_train_dataset.select(range(min(max_count, len(filtered_train_dataset))))
    test_dataset = filtered_test_dataset.select(range(min(max_count, len(filtered_test_dataset))))
    
    # Shuffle the datasets
    train_dataset = train_dataset.shuffle(seed=seed)
    test_dataset = test_dataset.shuffle(seed=seed)
    
    return train_dataset, test_dataset