"""
restrict output tokens using lm-format-enforcer

https://github.com/ServiceNow/context-is-key-forecasting/blob/main/cik_benchmark/baselines/direct_prompt.py#L99C5-L112C69

example output taken from direct_prompt

<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>
"""

import logging
import pandas as pd
import re

log = logging.getLogger(__name__)


def format_timeseries_input(df: pd.DataFrame, forecast_split: float) -> str:
    """
    Formats and returns the timeseries history used by the model to forecast
    unseen timestamp values.
    """

    if "date" not in df.columns or "value" not in df.columns:
        raise ValueError("DataFrame must contain 'date' and 'value' columns.")
    if not (0.05 < forecast_split < 0.95):
        raise ValueError("train_split must be a float between 0.05 and 0.95")

    # Ensure sorted by date
    df = df.sort_values(by="date", ascending=True)

    # Ensure the 'date' column is in daily 'YYYY-MM-DD' format
    try:
        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d").dt.strftime(
            "%Y-%m-%d"
        )
    except Exception as e:
        raise ValueError(f"Date column contains invalid formats: {e}")

    # Split data for forecasting
    split = 1 - int(len(df) * forecast_split)
    history = df.iloc[:split]
    forecast = df.iloc[split:]

    history_formatted = "\n".join(
        f"({row['date']}, {row['value']})" for _, row in history.iterrows()
    )
    history_section = f"<history>\n{history_formatted}\n</history>"

    forecast_formatted = "\n".join(
        f"({row['date']}, x)" for _, row in forecast.iterrows()
    )
    forecast_section = f"<forecast>\n{forecast_formatted}\n</forecast>"

    return f"{history_section}\n{forecast_section}"


def format_output_regex(timestamps):
    """
    Regex-enforced model output using a list of timestamps.
    """
    timestamp_regex = "".join(
        [
            r"\(\s*{}\s*,\s*[-+]?\d+(\.\d+)?\)\n".format(re.escape(ts))
            for ts in timestamps
        ]
    )
    return r"<forecast>\n{}<\/forecast>".format(timestamp_regex)
