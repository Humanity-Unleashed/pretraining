import os
import warnings
import logging
from humun_benchmark.utils.globals import ENV_VARS
from humun_benchmark.utils.read import read_env

log = logging.getLogger(__name__)


# audit and warn for missing environment variables
def check_env(vars=ENV_VARS, fill_missing=True):
    """
    Checks if env variables are set and issues a warning if missing.
    If `fill_missing` is true it will set the missing variables.
    """
    missing = [var for var in vars if os.getenv(var) is None]

    if fill_missing and missing:
        read_env()
        missing = [var for var in vars if os.getenv(var) is None]  # Recheck

    if missing:
        warnings.warn(
            f"Missing environment variables: {missing}. Check your setup.",
            UserWarning,
        )

    return {var: os.getenv(var) for var in vars}


def check_config():
    """
    checks configuration variables are all present. Warn if missing, log if some are replaced.
    """
    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
    )
    for var in ENV_VARS:
        log.info(f"removing var that exists: {var}")
        os.environ.pop(var, None)  # Remove variable if it exists

    check_env()
