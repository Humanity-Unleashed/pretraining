"""
Parent class for model interfaces.

"""

import logging
from typing import Any
from humun_benchmark.prompt import Prompt


log = logging.getLogger(__name__)


# parent class for all models
class Model:
    def __init__(self, label: str):
        self.label = label
        self._load_model()

    def _load_model(self):
        raise NotImplementedError("'_load_model' must be implemented in subclass.")

    def inference(
        self, prompt_instance: Prompt, n_runs: int, temperature: float
    ) -> None:
        """
        in-place method that appends inference responses to Prompt instance
        """
        raise NotImplementedError("`inference` must be implemented in subclass.")
