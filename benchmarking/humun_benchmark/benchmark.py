import pandas as pd
import json
import logging
from humun_benchmark.utils.checks import check_env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


if __name__ == "__main__":
    # check and fill in environment variables
    env_vars = check_env()

    # read in data

    # read in config for the run
    # - take both config and flags, flags override config (but log over-riding)

    # run checks on models / data compatability

    # run inference for chosen models
    # - instruct models
    # - ARIMA
    # - ETS, etc.

    # calculate metrics for all results
    # - CRPS, Rank, MAE etc.

    # produce table presenting results
