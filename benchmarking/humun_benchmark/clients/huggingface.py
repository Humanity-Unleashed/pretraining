import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from humun_benchmark.model import Model
from humun_benchmark.utils.errors import ModelError
from humun_benchmark.prompt import InstructPrompt
import logging

log = logging.getLogger(__name__)


class HuggingFace(Model):
    """
    Requires pre-downloaded models. Use `Make model <repo>` to download via huggingface cli.
    """

    def _load_model(self):
        try:
            log.info(f"Loading model from Hugging Face repo: {self.location}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.location, torch_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.location, torch_dtype=torch.float16
            )

            model.eval()
            model.to("cuda")
            self.tokenizer = tokenizer
            return model
        except Exception as e:
            raise ModelError(f"Failed to load model from Hugging Face: {str(e)}")

    def _format_response(self, response: torch.Tensor) -> str:
        return self.tokenizer.decode(response[0], skip_special_tokens=True)

    def inference(self, payload: InstructPrompt) -> str:
        try:
            inputs = self.tokenizer(payload.prompt_text, return_tensors="pt").to("cuda")
            output_tokens = self.model.generate(
                inputs["input_ids"], max_length=5000, temperature=1
            )
            return self._format_response(output_tokens)
        except Exception as e:
            raise ModelError(f"Inference failed: {str(e)}") from e
