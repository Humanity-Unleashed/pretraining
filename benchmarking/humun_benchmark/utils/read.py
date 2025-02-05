import json
import logging
import os
from typing import Dict, List

import pandas as pd

log = logging.getLogger(__name__)


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
