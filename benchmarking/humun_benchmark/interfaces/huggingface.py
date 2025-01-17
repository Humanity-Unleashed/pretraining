"""
HF interface usage heavily inspired by CiK:
    - https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/hf_utils/dp_hf_api.py

LMs configured with Flash-Attention-2 for efficiency:
    - https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2
"""

import torch
import gc
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, pipeline
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from humun_benchmark.model import Model
from humun_benchmark.utils.errors import ModelError, ModelLoadError
from humun_benchmark.utils.parse import parse_forecast_output
from humun_benchmark.prompt import InstructPrompt
from types import SimpleNamespace
import logging

log = logging.getLogger(__name__)

LLM_Map = {
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "ministral-8b-instruct-2410": "mistralai/Ministral-8B-Instruct-2410",
}

DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def path_from_map(llm):
    """
    Returns LLM repo via mapping or logs that it's not recognised.
    """
    if llm in LLM_Map:
        return LLM_Map[llm]

    # attempt mapping by assuming repo name was provided
    attempt = llm.split("/")[-1].tolower()
    if attempt in LLM_Map[attempt]:
        return LLM_Map[attempt]

    log.info("LLM name provided not in `LLM_Map` mapping.")
    return llm


def get_model_and_tokenizer(llm: str):
    """
    Returns tokenizer and model from HF interface.

    Params:
        llm (str): model name
    """

    llm_path = path_from_map(llm)

    try:
        log.info(f"Loading model from Hugging Face repo: {llm_path}")

        # can make custom configs per language model via if-elses
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, padding_side="left", torch_dtype=torch.float16
        )
        model = LlamaForCausalLM.from_pretrained(
            llm_path,
            # load everything to one GPU - peer mapping currently not configured
            device_map={"": "cuda:4"},
            torch_dtype=torch.float16,
        )
        model.eval()

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
        return model, tokenizer

    except Exception as e:
        raise ModelLoadError(f"Failed to load model from Hugging Face: {e}")


class HuggingFace(Model):
    """
    Configures and handles Hugging Face LLM inference.

    Contains logic from CiK's DirectPrompt"""

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
            device_map={"": "cuda:4"},  # device mapping issues on server
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
        choices = []
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

    # batch/parallel inference method below -
    #
    # [AL:] I believe the GPU server is not configured to run things in batch at the moment.
    # Errors received include:
    # RuntimeError: CUDA error: peer mapping resources exhausted.
    # Looking at `top` I noticed a few other people working on the server, though their processes weren't occupying GPU space, instead CPU. Likely just not configured yet (e.g, nvcc exists but path isn't available to users.)
    #
    # This thought is reinforced by the inability to install flash-attention as it doesn't have access
    # to `nvcc` to compile kernels.
    #
    #
    #
    # @torch.inference_mode()
    # def inference(
    #     self, payload: InstructPrompt, n_runs: int = 1, temperature: float = 1
    # ):
    #     if not 0 < n_runs <= 10:
    #         raise ModelError(f"Improper number of inference runs: {n_runs}")
    #
    #     # initialise for clean up later
    #     batch_inputs, output_tokens = None, None
    #
    #     try:
    #         # prepare batch inputs
    #         inputs = self.tokenizer(payload.prompt_text, return_tensors="pt")
    #         batch_inputs = {k: v.repeat(n_runs, 1) for k, v in inputs.items()}
    #         batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
    #
    #         log.info(f"Running inference in parallel for {n_runs} runs.")
    #
    #         output_tokens = self.model.generate(
    #             **batch_inputs,
    #             max_length=10000,
    #             temperature=temperature,
    #             do_sample=True,
    #         )
    #
    #         # decode and store responses
    #         formatted_responses = self.tokenizer.batch_decode(
    #             output_tokens, skip_special_tokens=True
    #         )
    #         payload.responses.extend(formatted_responses)
    #
    #     except Exception as e:
    #         raise ModelError(f"Inference failed: {str(e)}") from e
    #
    #     finally:
    #         # cleanup GPU memory
    #         del batch_inputs
    #         del output_tokens
    #         gc.collect()
    #         torch.cuda.empty_cache()
