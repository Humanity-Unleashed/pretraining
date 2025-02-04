"""
Parent class for model interfaces.

"""

import logging
from humun_benchmark.prompt import Prompt

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, label: str):
        self.label = label
        self._load_model()

    def _load_model(self):
        logger.exception("'_load_model' must be implemented in subclass.")
        raise NotImplementedError("'_load_model' must be implemented in subclass.")

    def inference(
        self, prompt_instance: Prompt, n_runs: int, temperature: float
    ) -> None:
        """
        in-place method that appends inference responses to Prompt instance
        """
        logger.exception("`inference` must be implemented in subclass.")
        raise NotImplementedError("`inference` must be implemented in subclass.")

    def serialise():
        """
        Write out model configuration for logs.
        """
        logger.exception("`serialise` must be implemented in subclass.")
        raise NotImplementedError("`serialise` must be implemented in subclass.")
