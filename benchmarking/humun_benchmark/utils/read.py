import json
import logging
import os
from typing import Dict, List
import argparse

import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)


def read_parquet(file_path: str):
    """
    Detailed analysis of parquet file structure and contents.
    """
    parquet_file = pq.ParquetFile(file_path)

    # Basic file info
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    log.info(f"\nFile size: {file_size_gb:.2f} GB")

    # Schema details
    log.info("\nDetailed Schema:")
    log.info(str(parquet_file.schema))

    # Metadata
    metadata = parquet_file.metadata
    log.info(f"\nNumber of rows: {metadata.num_rows}")
    log.info(f"Number of columns: {metadata.num_columns}")
    log.info(f"Number of row groups: {metadata.num_row_groups}")

    # Row group details
    log.info("\nRow Group Details:")
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        log.info(f"\nRow Group {i}:")
        log.info(f"Number of rows: {row_group.num_rows}")
        log.info(f"Total byte size: {row_group.total_byte_size / (1024**2):.2f} MB")

        # Column details within row group
        log.info("\nColumn Details:")
        for j in range(row_group.num_columns):
            col = row_group.column(j)
            log.info(f"\nColumn {j} ({col.path_in_schema}):")
            log.info(f"Type: {col.physical_type}")
            log.info(f"Compressed size: {col.total_compressed_size / 1024:.2f} KB")
            log.info(f"Uncompressed size: {col.total_uncompressed_size / 1024:.2f} KB")

    # Read a sample to understand structure
    df_sample = pd.read_parquet(file_path, nrows=5)
    log.info("\nSample Data Types:")
    log.info(df_sample.dtypes)

    # Get unique values in series_id if it exists
    if "series_id" in df_sample.columns:
        n_series = len(df_sample["series_id"].unique())
        log.info(f"\nNumber of unique series in sample: {n_series}")

    return parquet_file


def read_data(file_paths: List) -> Dict[str, pd.DataFrame]:
    """
    reads data from a list of csv/json file paths and returns a dictionary of DataFrames.
    """
    dataframes = {}
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            log.warning(f"Unsupported file format: {file_path}. Skipping.")
            continue

        dataframes[os.path.basename(file_path)] = df

    return dataframes


def read_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        log.exception(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = json.load(file)

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="read parquet details")
    parser.add_argument(
        "--filepath",
        type=str,
        default="/workspace/datasets/fred/split.parquet",
    )

    args = parser.parse_args()
    read_parquet(args.filepath)
