import argparse
import json
import logging
import os
from datetime import datetime
from pprint import pformat

import pandas as pd

from humun_benchmark.interfaces.huggingface import HuggingFace
from humun_benchmark.prompt import InstructPrompt
from humun_benchmark.utils.checks import check_env
from humun_benchmark.utils.logging import setup_logging
from humun_benchmark.utils.tasks import NUMERICAL

# load .env and check needed variables exist
check_env()

# timestamp for output files
time = datetime.now().strftime("%Y%m%d_%H%M%S")


def generate(
    datasets: list[str] = ["data/fred/test.csv"],
    model: str = "llama-3.1-8b-instruct",
    output_path: str = os.getenv("RESULTS_STORE"),
    batch_size: int = 1,
) -> None:
    """
    Generate forecasts from a model and save results.
    """

    # make a timestep folder for outputs to be written to
    output_path = output_path or os.getenv("RESULTS_STORE")
    output_path = os.path.join(output_path, time)
    # make sure it exists
    os.makedirs(output_path, exist_ok=True)

    # sets up logging config and gives a filename
    setup_logging(f"{output_path}/benchmark.log")
    log = logging.getLogger("humun_benchmark.generate")

    params = {
        "datasets": datasets,
        "model": model,
        "output_path": output_path,
        "batch_size": batch_size,
    }
    log.info(f"Run Parameters:\n{pformat(params)}")

    # create model instance and log config
    llm = HuggingFace(model)
    log.info(f"Model Info:\n{pformat(llm.serialise())}")

    results = {}
    for dataset in datasets:
        # load input data
        timeseries_df = pd.read_csv(dataset)

        # create prompt instance
        prompt = InstructPrompt(task=NUMERICAL, timeseries=timeseries_df)

        # run inference
        llm.inference(payload=prompt, n_runs=batch_size)

        # store results
        results[dataset] = prompt.results_df.to_json(
            orient="records", date_format="iso"
        )

    if results:
        json_path = f"{output_path}/{llm.label}.json"
        with open(json_path, "w") as f:
            json.dump(results, f)
        log.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model outputs using a Hugging Face LLM."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["data/fred/test.csv"],
        help="Path/s to the input CSV file. Default: [data/fred/test.csv]",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.1-8b-instruct",
        help="Name of the Hugging Face model to use. Default: llama-3.1-8b-instruct",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.getenv("RESULTS_STORE"),
        help="Directory to save the output pickle files. Default: workspace/pretraining/benchmarks/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of runs for inference (n_runs). Default: 1",
    )
    args = parser.parse_args()
    generate(**vars(args))
