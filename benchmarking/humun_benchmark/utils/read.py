import json
import logging
from typing import List, Dict
import pandas as pd
import os
import warnings
from dotenv import dotenv_values, set_key
from humun_benchmark.utils.globals import ENV_VARS

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
            warnings.warn(
                f"Unsupported file format: {file_path}. Skipping.", UserWarning
            )
            continue

        dataframes[os.path.basename(file_path)] = df

    return dataframes


def read_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        config = json.load(file)

    return config


def read_env(env_path=".env"):
    """
    reads in default .env file and replaces on system if non-existent.
    """
    if not os.path.exists(env_path):
        log.warning(f"{env_path} not found. Skipping environment setup.")
        return

    env_values = dotenv_values(env_path)
    replaced = []

    for key, value in env_values.items():
        if key not in os.environ:
            os.environ[key] = value  # Properly set the environment variable
            set_key(env_path, key, value)  # Persist in the .env file
            replaced.append(key)

    if replaced:
        log.info(
            "Environment variables replaced that were missing:\n"
            + "\n".join(f"* {var}" for var in replaced)
        )

    # Debugging: Print the values of environment variables
    for var in ENV_VARS:
        log.info(f"{var} = {os.getenv(var)}")
