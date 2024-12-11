from typing import Dict, Any


class ModelError(Exception):
    pass


class InferenceError(ModelError):
    pass


class ModelLoadError(ModelError):
    pass


class APIStatusError(ModelError):
    def __init__(self, status: int, response: Dict[str, Any]):
        self.status = status
        self.response = response
        super().__init__(f"Status returned: {status}")
