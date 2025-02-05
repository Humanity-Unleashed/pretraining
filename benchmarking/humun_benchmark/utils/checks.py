import logging
import os

from dotenv import load_dotenv

log = logging.getLogger(__name__)

ENV_VARS = ["DATA_STORE", "RESULTS_STORE", "HF_HOME", "HF_TOKEN_PATH"]


def check_env(env_path=".env", vars=ENV_VARS):
    """
    Loads environment variables from dotenv and checks in any are missing, issuing a warning.
    """
    if not load_dotenv(env_path):
        log.warning(f"{env_path} not found. Skipping environment setup.")

    missing = [var for var in vars if os.getenv(var) is None]
    if missing:
        log.warning(f"Missing environment variables: {missing}. Check your setup.")


def check_config():
    """
    checks configuration variables are all present. Warn if missing, log if some are replaced.
    """
    return None
