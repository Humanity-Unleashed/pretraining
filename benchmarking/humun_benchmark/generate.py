import argparse
import pandas as pd
import os
from datetime import datetime
import torch
import logging
import pickle
from pprint import pformat

from humun_benchmark.interfaces.huggingface import HuggingFace
from humun_benchmark.utils.tasks import NUMERICAL
from humun_benchmark.prompt import InstructPrompt
from humun_benchmark.utils.checks import check_env
from humun_benchmark.utils.logging import setup_logging

# Load .env and check needed variables exist
check_env()

# Timestamp for output files
time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Sets up logging config and give a filename
setup_logging(f"{os.getenv('RESULTS_STORE')}/{time}.log")

log = logging.getLogger("humun_benchmark.generate")

"""
To log:

* model / inference config
* datasets used
* time taken for inference
* hardware set up
* pyspy or similar for hardware usage stats over time
* resulting metrics

"""


def main(args):
    """
    Note: Currently only handles one timeseries to run inference on.
        - pd.DataFrame with columns ['date', 'value']

    Turn into generate() which can be passed args individually from __main__, or called from another program. This would allow for a per-model inference to be generated.
    """

    # Ensure the output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    log.info(f"Ensured the output directory exists: {args.output_path}")

    # Check if GPU is available
    if torch.cuda.is_available():
        log.info("CUDA is available.")
    else:
        log.warning("CUDA is not available.")

    # Create model instance
    llm = HuggingFace(args.model)
    log.info(f"Model instance created for model: {args.model}")

    # Load input data
    timeseries_df = pd.read_csv(args.input_path)
    log.info(f"Input data loaded from: {args.input_path}")

    # Create prompt instance
    prompt = InstructPrompt(task=NUMERICAL, timeseries=timeseries_df)
    log.info("Prompt instance created.")

    # Run inference
    llm.inference(payload=prompt, n_runs=args.batch_size)

    # Serialise the model to logs
    # TODO: add model results conditionally in serialise()
    model_info = llm.serialise()
    log.info(f"Model Info:\n {pformat(model_info)}")

    # Save the prompt with responses as a pickle file
    save_file = os.path.join(args.output_path, f"{time}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(prompt, f)

    log.info(f"Prompt saved to {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate model outputs using a Hugging Face LLM."
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="data/fred/test.csv",
        help="Path to the input CSV file. Default: data/fred/test.csv",
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

    # Parse the arguments and run the main function
    args = parser.parse_args()

    main(args)
