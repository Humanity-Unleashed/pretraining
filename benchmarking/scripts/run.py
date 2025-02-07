"""
Generic script to run a number of benchmarks.

Requirements:
* action required: download all_fred_metadata.csv using humun_benchmark/adhoc/downloadGC.py
    * see file for API key instructions.
"""

from humun_benchmark.benchmark import benchmark

METADATA_PATH = "all_fred_metadata.csv"  # CHANGE THIS IF DIFFERENT!!

DATASETS_PATH = "/workspace/datasets/fred/split.parquet"
RESULTS_STORE = "."  # will save to a datestamped folder

# config variables manually set, feel free to change.
# note: around batch_size = 25 & n_datasets >= 3 I started to encounter CUDA mem issues.
N_DATASETS = 3
BATCH_SIZE = 15

MODELS = [
    "llama-3.1-8b-instruct",
    "Qwen2.5-7B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
]

benchmark(
    datasets_path=DATASETS_PATH,
    metadata_path=METADATA_PATH,
    output_path=RESULTS_STORE,
    selector={"frequency": "Monthly"},
    n_datasets=N_DATASETS,
    models=MODELS,
    batch_size=BATCH_SIZE,
)
