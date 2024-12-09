from typing import Dict, Any, Optional, Union
from pydantic import BaseModel
import pandas as pd
from humun_benchmark.utils.format import format_timeseries_input


# Parent class for all prompt methods
class Prompt(BaseModel):
    task: str
    timeseries: pd.DataFrame
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    prompt_text: Optional[str] = None
    responses: Optional[Union[Dict[str, Any], str]] = None

    class Config:
        arbitrary_types_allowed = True  # Allow pd.DataFrame as a field


class InstructPrompt(Prompt):
    def __init__(
        self,
        task: str,
        timeseries: pd.DataFrame,
        context: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # calls pydantic constructor to initialise fields
        super().__init__(
            task=task,
            timeseries=timeseries,
            context=context,
            metadata=metadata,
            prompt_text=self._format_input(task, timeseries, context, metadata),
            responses="Inference not run yet.",
        )

    @staticmethod
    def _format_input(
        task: str,
        timeseries: pd.DataFrame,
        context: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """
        Returns formatted input text.
        """
        prompt_text = task

        if context:
            prompt_text += f"<context>\n{context}\n</context>\n"
        if metadata:
            prompt_text += f"<metadata>\n{metadata}\n</metadata>\n"

        prompt_text += format_timeseries_input(timeseries, forecast_split=0.2)
        return prompt_text


class MultiModalPrompt(Prompt):
    """
    Prompt for multimodal models.
    """

    pass
