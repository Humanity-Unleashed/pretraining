"""
Parent class for model interfaces.

"""

import logging
from typing import Any
from humun_benchmark.prompt import Prompt


log = logging.getLogger(__name__)


# parent class for all models
class Model:
    def __init__(self, label: str, location: str):
        self.label = label
        self.location = location  # local path or huggingface repo ID
        self.model = self._load_model()

    def _load_model(self):
        raise NotImplementedError("'_load_model' must be implemented in subclass.")

    def _format_response(self, response: Any) -> str:
        raise NotImplementedError("`_format_response` must be implemented in subclass.")

    def inference(self, prompt_instance: Prompt) -> Any:
        raise NotImplementedError("`inference` must be implemented in subclass.")
