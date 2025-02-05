from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from humun_benchmark.utils.format import format_timeseries_input


# Parent class for all prompt methods
class Prompt(BaseModel):
    task: str
    timeseries: pd.DataFrame
    forecast_split: float = 0.2
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    prompt_text: Optional[str] = None
    responses: List[str] = []

    class Config:
        arbitrary_types_allowed = True  # Allow pd.DataFrame as a field

    def serialise(self) -> Dict[str, Any]:
        """
        Serialize prompt configuration and data for logging.
        """
        info = (
            {
                "task": self.task,
                "forecast_split": self.forecast_split,
                "context": self.context,
                "metadata": self.metadata,
                "forecasts": len(self.responses),
            },
        )

        return info


class InstructPrompt(Prompt):
    results_df: Optional[pd.DataFrame] = None

    def __init__(
        self,
        task: str,
        timeseries: pd.DataFrame,
        forecast_split: float = 0.2,
        context: str = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # calls pydantic constructor to initialise fields
        super().__init__(
            task=task,
            timeseries=timeseries,
            forecast_split=forecast_split,
            context=context,
            metadata=metadata,
        )
        self.prompt_text = self._format_input()

    def _format_input(self) -> str:
        """
        Returns formatted input text.
        """
        prompt_text = self.task

        if self.context:
            prompt_text += f"<context>\n{self.context}\n</context>\n"
        if self.metadata:
            prompt_text += f"<metadata>\n{self.metadata}\n</metadata>\n"

        prompt_text += format_timeseries_input(self.timeseries, self.forecast_split)
        return prompt_text

    def merge_forecasts(self, dfs: List[pd.DataFrame]):
        """
        Merge forecast responses together
        """
        # Rename the value columns to forecast_1, forecast_2, ..., forecast_n
        for i, df in enumerate(dfs, start=1):
            df.rename(columns={"value": f"forecast_{i}"}, inplace=True)

        # Merge all dataframes on the date column
        merged_df = dfs[0]

        if len(dfs) > 1:
            for df in dfs[1:]:
                merged_df = pd.merge(merged_df, df, on="date", how="outer")

        self.results_df = merged_df


class MultiModalPrompt(Prompt):
    """
    Prompt for multimodal models.
    """

    pass
