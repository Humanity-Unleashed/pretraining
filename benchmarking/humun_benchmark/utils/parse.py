import pandas as pd
import re


def parse_forecast_output(response: str) -> pd.DataFrame:
    # section regex pattern
    history_pattern = r"<history>\n(.*?)</history>"
    forecast_pattern = r"<forecast>\n(.*?)</forecast>"

    # get history and forecast sections
    history_matches = re.search(history_pattern, response, re.DOTALL)
    forecast_matches = re.search(forecast_pattern, response, re.DOTALL)

    def parse_section(section: str):
        if not section:
            return []
        # date-value regex pattern
        data_matches = re.findall(r"\(([\d\-]+),\s*([\d.]+)\)", section)
        return [(date, float(value)) for date, value in data_matches]

    # parse each section
    history_data = parse_section(history_matches.group(1) if history_matches else "")
    forecast_data = parse_section(forecast_matches.group(1) if forecast_matches else "")

    # return as dataframe
    combined_data = history_data + forecast_data
    return pd.DataFrame(combined_data, columns=["date", "value"])
