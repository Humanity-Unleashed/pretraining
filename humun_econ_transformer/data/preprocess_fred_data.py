"""
Loads the `all_FRED_merged.csv` file and chunks the data based on flexible prediction windows.

Key features:
- Variable prediction window length (randomly selected between 2 and max_prediction_window)
- Context window determined by context_multiplier * prediction_window
- Optional date filtering with max_cutoff_year
- Each data point is a tuple of ('YYYY-MM-DD', value)

The following two parquet files are saved:
`fred_train.parquet`: All but the last chunk of each time series, for training.
`fred_test.parquet`: The last chunk of each time series, for evaluation.
"""
import argparse
import pandas as pd
import ast
import os
import random
from datetime import datetime
from tqdm import tqdm

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

def main():
    parser = argparse.ArgumentParser(description="Process FRED dataset into training and test sets with flexible windows.")
    parser.add_argument("--dataset_path", type=str, help="Path to the input CSV dataset")
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata CSV")
    parser.add_argument("--output_folder", type=str, help="Folder to save output Parquet files")
    parser.add_argument("--max_prediction_window", type=int, default=6, 
                        help="Maximum number of values to predict (actual window size will be randomly chosen between 2 and this value)")
    parser.add_argument("--context_multiplier", type=int, default=7, 
                        help="Multiplier to determine context window size (context_window = context_multiplier * prediction_window)")
    parser.add_argument("--num_eval_chunks", type=int, default=1, 
                        help="Number of chunks to use for evaluation")
    parser.add_argument("--max_cutoff_year", type=int, default=None, 
                        help="Only consider data points from this year onwards (optional)")
    parser.add_argument("--random_seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.random_seed)
    
    df = pd.read_csv(args.dataset_path)
    
    def safe_literal_eval(x):
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError) as e:
                print(f"Skipping invalid entry: {x} - Error: {e}")
                return None  # Return original value if parsing fails
        return x

    df['data'] = df['data'].apply(safe_literal_eval)
    df = df.dropna(subset=['data'])

    train_data, test_data = [], []
    skipped_series = 0
    processed_series = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        train_chunks, test_chunks = process_series(
            row['series_id'], 
            row['title'], 
            row['data'], 
            args.max_prediction_window, 
            args.context_multiplier, 
            args.num_eval_chunks, 
            args.max_cutoff_year
        )
        
        if not train_chunks and not test_chunks:
            skipped_series += 1
            continue
            
        processed_series += 1
        
        for history, forecast in train_chunks:
            train_data.append((row['series_id'], row['title'], history, forecast))
        
        for history, forecast in test_chunks:
            test_data.append((row['series_id'], row['title'], history, forecast))
    
    # Convert to DataFrame
    train_df = pd.DataFrame(train_data, columns=['series_id', 'title', 'history', 'forecast'])
    test_df = pd.DataFrame(test_data, columns=['series_id', 'title', 'history', 'forecast'])
    
    # Add statistics
    print(f"Processed {processed_series} time series, skipped {skipped_series} time series")
    print(f"Created {len(train_df)} training examples and {len(test_df)} test examples")
    
    # Calculate average prediction and context window sizes
    if len(train_df) > 0:
        avg_pred_train = sum(len(forecast) for forecast in train_df['forecast']) / len(train_df)
        avg_ctx_train = sum(len(history) for history in train_df['history']) / len(train_df)
        print(f"Training set - Avg prediction window: {avg_pred_train:.2f}, Avg context window: {avg_ctx_train:.2f}")
    
    if len(test_df) > 0:
        avg_pred_test = sum(len(forecast) for forecast in test_df['forecast']) / len(test_df)
        avg_ctx_test = sum(len(history) for history in test_df['history']) / len(test_df)
        print(f"Test set - Avg prediction window: {avg_pred_test:.2f}, Avg context window: {avg_ctx_test:.2f}")
    
    # Save to parquet files
    os.makedirs(args.output_folder, exist_ok=True)
    train_path = os.path.join(args.output_folder, "fred_train.parquet")
    test_path = os.path.join(args.output_folder, "fred_test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved training data to {train_path} with {len(train_df)} examples.")
    print(f"Saved test data to {test_path} with {len(test_df)} examples.")

if __name__ == "__main__":
    main()