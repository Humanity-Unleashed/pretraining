import pandas as pd
from humun_benchmark.clients.huggingface import HuggingFace
from humun_benchmark.utils.tasks import NUMERICAL
from humun_benchmark.prompt import InstructPrompt

TEST_CSV = "data/fred/test.csv"
TEST_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

if __name__ == "__main__":
    # create model instance
    llm = HuggingFace("llama-3.1-8B-Instruct", TEST_MODEL)

    # create prompt instance
    timeseries_df = pd.read_csv(TEST_CSV)
    prompt = InstructPrompt(task=NUMERICAL, timeseries=timeseries_df)

    print("running inference")

    # run queries
    response = llm.inference(payload=prompt)

    print(f"Response:\n{response}")
