"""
HF interface usage heavily inspired by CiK:
    - https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/hf_utils/dp_hf_api.py

LMs configured with Flash-Attention-2 for efficiency:
    - https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
"""

import logging
import re

import torch
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    pipeline,
    AutoModelForCausalLM,
)

from humun_benchmark.model import Model
from humun_benchmark.prompt import InstructPrompt
from humun_benchmark.utils.errors import ModelError, ModelLoadError
from humun_benchmark.utils.parse import parse_forecast_output

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


# some models can be retrieved with just their name instead of repo
LLM_Map = {
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "ministral-8b-instruct-2410": "mistralai/Ministral-8B-Instruct-2410",
}

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_model_and_tokenizer(llm):
    """
    Returns

    """
    # check if repo has been provided in model name
    if "/" not in llm:
        if "llama" in llm:
            llm = "meta-llama/" + llm
        elif "istral" in llm:
            llm = "mistralai/" + llm
        elif "qwen" in llm:
            llm = "Qwen/"

    # case-specific model/tokenizer loaders
    try:
        if "llama-3" in llm:
            tokenizer = AutoTokenizer.from_pretrained(
                llm, padding_side="left", legacy=False
            )
            model = LlamaForCausalLM.from_pretrained(
                llm,
                device_map="auto",  # {"": "cuda:4"},
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                llm,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        # configure special tokens
        special_tokens_dict = dict()
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.pad_token = tokenizer.eos_token

        log.info("Model and tokenizer loaded successfully.")

    except Exception as e:
        raise ModelLoadError(f"Failed to load model from Hugging Face: {e}")

    return model.eval(), tokenizer


class HuggingFace(Model):
    """
    Configures and handles Hugging Face LLM inference.
    """

    def _load_model(self):
        self.model, self.tokenizer = get_model_and_tokenizer(self.label)

    @torch.inference_mode()
    def inference(
        self,
        payload: InstructPrompt,
        n_runs=1,
        temperature=1.0,
        constrained_decoding=True,
        **kwargs,
    ):
        """Appends inference output to `payload.responses`"""

        if not 0 < n_runs <= 20:
            raise ModelError(f"Improper number of inference runs: {n_runs}")

        if constrained_decoding:
            # get future timestamps
            split = 1 - int(len(payload.timeseries) * payload.forecast_split)
            future_timestamps = payload.timeseries.iloc[split:]["date"]

        def constrained_decoding_regex(required_timestamps):
            """
            Generates a regular expression to force the model output
            to satisfy the required format and provide values for
            all required timestamps
            """
            timestamp_regex = "".join(
                [
                    r"\(\s*{}\s*,\s*[-+]?\d+(\.\d+)?\)\n".format(re.escape(ts))
                    for ts in required_timestamps
                ]
            )
            return r"<forecast>\n{}<\/forecast>".format(timestamp_regex)

        # Make generation pipeline
        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",  # {"": "cuda:4"},  # device mapping issues on server
        )

        # Build a regex parser with the generated regex
        parser = RegexParser(constrained_decoding_regex(future_timestamps))
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipe.tokenizer, parser
        )

        # info for debugging cuda issues
        log.info(
            f"Running inference sequentially on {self.model.device} for {n_runs} time/s."
        )

        # Now extract the assistant's reply
        for response in pipe(
            [payload.prompt_text] * n_runs,
            max_length=10000,
            temperature=temperature,
            prefix_allowed_tokens_fn=prefix_function,
            batch_size=n_runs,
        ):
            output_start = len(payload.prompt_text)
            forecast_output = response[0]["generated_text"][output_start:]
            payload.responses.append(forecast_output)

        # turn text response/s into a results dataframe
        dfs = [parse_forecast_output(df) for df in payload.responses]
        payload.merge_forecasts(dfs)  # sets payload.results_df in-place
