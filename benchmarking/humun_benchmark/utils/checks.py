import logging
import os
from dotenv import load_dotenv

log = logging.getLogger(__name__)

ENV_VARS = [
    "DATASETS_PATH",
    "METADATA_PATH",
    "RESULTS_STORE",
    "HF_HOME",
    "HF_TOKEN_PATH",
]


def setup_env(env_path=".env"):
    """
    Loads environment variables from dotenv file.
    Returns True if successful, False if .env not found.
    """
    if not load_dotenv(env_path):
        log.warning(f"{env_path} not found. Skipping environment setup.")
        return False
    return True


def check_env(vars=ENV_VARS):
    """
    Checks if any required environment variables are missing.
    Logs a warning if any are missing.
    """
    missing = [var for var in vars if os.getenv(var) is None]
    if missing:
        log.warning(f"Missing environment variables: {missing}. Check your setup.")
        return False
    return True


# Load environment variables at module import
setup_env()
