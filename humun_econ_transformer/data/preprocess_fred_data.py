"""
Loads the `all_FRED_merged.csv` file and chunks the data given a `context_window` and `prediction_window`.
Each chunk is disjoint of length `prediction_window + context_window`, where the `history` column contains
the first `context_window` values and the `forecast` column contains the remaining `prediction_window` values.
The following two parquet files are saved:

`fred_train.parquet`: All but the last chunk of each time series, for training.
`fred_test.parquet`: The last chunk of each time series, for evaluation.
"""
import argparse
import pandas as pd
import ast
import os
from tqdm import tqdm

def process_series(series_id, title, data_dict, prediction_window, context_window, num_eval_chunks):
    sorted_data = sorted(data_dict.items(), key=lambda x: x[0])
    chunk_size = prediction_window + context_window
    chunks = [sorted_data[i:i + chunk_size] for i in range(0, len(sorted_data), chunk_size)]
    valid_chunks = [chunk for chunk in chunks if all(value is not None and value != '.' for _, value in chunk)]
    
    num_eval_chunks = min(num_eval_chunks, len(valid_chunks) - 1)
    if num_eval_chunks <= 0:
        return valid_chunks, [] # If there's only one chunk, put it in training

    train_chunks = valid_chunks[:-num_eval_chunks]
    test_chunks = valid_chunks[-num_eval_chunks:]
    
    return train_chunks, test_chunks

def main():
    parser = argparse.ArgumentParser(description="Process FRED dataset into training and test sets.")
    parser.add_argument("--dataset_path", type=str, help="Path to the input CSV dataset")
    parser.add_argument("--metadata_path", type=str, help="Path to the metadata CSV")
    parser.add_argument("--output_folder", type=str, help="Folder to save output Parquet files")
    parser.add_argument("--context_window", type=int, default=42, help="Number of history values per chunk")
    parser.add_argument("--prediction_window", type=int, default=6, help="Number of forecast values per chunk")
    parser.add_argument("--num_eval_chunks", type=int, default=1, help="Number of chunks to use for evaluation")
    
    args = parser.parse_args()
    
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
    
    for _, row in tqdm(df.iterrows()):
        train_chunks, test_chunks = process_series(
            row['series_id'], row['title'], row['data'], args.prediction_window, args.context_window, args.num_eval_chunks
        )
        
        for chunk in train_chunks:
            train_data.append((row['series_id'], row['title'], chunk[:args.context_window], chunk[args.context_window:]))
        
        for chunk in test_chunks:
            test_data.append((row['series_id'], row['title'], chunk[:args.context_window], chunk[args.context_window:]))
    
    train_df = pd.DataFrame(train_data, columns=['series_id', 'title', 'history', 'forecast'])
    test_df = pd.DataFrame(test_data, columns=['series_id', 'title', 'history', 'forecast'])
    
    os.makedirs(args.output_folder, exist_ok=True)
    train_path = os.path.join(args.output_folder, "fred_train.parquet")
    test_path = os.path.join(args.output_folder, "fred_test.parquet")
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"Saved training data to {train_path} with {len(train_df)} examples.")
    print(f"Saved test data to {test_path} with {len(test_df)} examples.")

if __name__ == "__main__":
    main()


