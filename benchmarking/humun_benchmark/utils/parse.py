import re

import pandas as pd

import logging

log = logging.getLogger(__name__)


def parse_forecast_output(response: str) -> pd.DataFrame:
    """Parse forecast output text into a DataFrame."""
    # Get forecast section
    forecast_pattern = r"<forecast>\n(.*?)</forecast>"
    forecast_match = re.search(forecast_pattern, response, re.DOTALL)

    if not forecast_match:
        log.info(f"Response: {response}")
        raise ValueError("No forecast section found in response")

    forecast_text = forecast_match.group(1)

    # Parse (date, value) pairs
    data_matches = re.findall(r"\(([\d\-\s:]+),\s*(-?[\d.]+)\)", forecast_text)

    if not data_matches:
        raise ValueError("No valid forecast data found in response")

    # Convert to DataFrame
    forecast_data = [(date.strip(), float(value)) for date, value in data_matches]
    df = pd.DataFrame(forecast_data, columns=["date", "value"])

    # Convert dates to datetime
    df["date"] = pd.to_datetime(df["date"])

    return df
