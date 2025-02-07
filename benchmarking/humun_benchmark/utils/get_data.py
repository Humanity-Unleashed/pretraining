import logging
from typing import Dict, List

import pandas as pd

log = logging.getLogger(__name__)


def get_data(
    n_datasets: int,
    metadata_path: str,
    datasets_path: str,
    filters: Dict = {"frequency": "Monthly"},  # Default to Monthly
) -> Dict[str, Dict]:
    """
    Get a random selection of n_datasets with both metadata and time series data.
    This function reads in extra candidate datasets (3x the number requested, if available)
    to act as backups in case some datasets have issues such as missing values.

    Args:
        n_datasets: Number of valid datasets to return.
        metadata_path: Path to metadata CSV file.
        datasets_path: Path to parquet file with time series data.
        filters: Dictionary of column:value pairs to filter metadata on.

    Returns:
        Dictionary of the format:
            {
                "series_id": {
                    "metadata": <metadata dict>,
                    "timeseries": <history data>
                },
                ...
            }

    Raises:
        ValueError: If there are fewer than n_datasets in the metadata, or no valid datasets
                    are found after processing.
    """
    # Read metadata and apply filters.
    metadata_df = pd.read_csv(metadata_path)

    for column, value in filters.items():
        if column in metadata_df.columns:
            # For string columns, do a case-insensitive comparison.
            if metadata_df[column].dtype == "object":
                metadata_df = metadata_df[
                    metadata_df[column].str.lower() == str(value).lower()
                ]
            else:
                metadata_df = metadata_df[metadata_df[column] == value]
        else:
            raise ValueError(f"Erroneous filter provided: {column}")

    if len(metadata_df) < n_datasets:
        raise ValueError(
            f"Only {len(metadata_df)} datasets available after filtering, requested {n_datasets}"
        )

    # Determine how many candidate datasets to sample.
    backup_multiplier = 3
    sample_size = min(backup_multiplier * n_datasets, len(metadata_df))
    selected_metadata = metadata_df.sample(n=sample_size)
    selected_ids = selected_metadata["id"].tolist()

    # Read time series data for all candidate IDs using parquet filtering.
    ts_df = pd.read_parquet(
        datasets_path,
        filters=[("series_id", "in", selected_ids)],  # Only read rows we need
    )

    # Process candidates until we have n_datasets valid ones.
    result = {}
    skipped_ids = []
    for series_id in selected_ids:
        # Stop processing if we have enough valid datasets.
        if len(result) >= n_datasets:
            break

        ts_series = ts_df[ts_df["series_id"] == series_id]
        if ts_series.empty:
            log.warning(f"No timeseries data found for series_id: {series_id}")
            skipped_ids.append(series_id)
            continue

        meta_series = selected_metadata[selected_metadata["id"] == series_id]
        if meta_series.empty:
            log.warning(f"No metadata found for series_id: {series_id}")
            skipped_ids.append(series_id)
            continue

        try:
            # Extract history data and check for missing values denoted by '.'
            history = ts_series.iloc[0].to_dict()["history"]
            if any(row[1] == "." for row in history):
                log.warning(
                    f"Series {series_id} contains missing values ('.'). Skipping."
                )
                skipped_ids.append(series_id)
                continue

            result[series_id] = {
                "metadata": meta_series.iloc[0].to_dict(),
                "timeseries": history,
            }
        except (KeyError, IndexError) as e:
            log.warning(f"Error processing series {series_id}: {str(e)}")
            skipped_ids.append(series_id)
            continue

    if not result:
        raise ValueError("No valid data found for any of the selected series")

    if len(result) < n_datasets:
        log.warning(
            f"Only found {len(result)} valid datasets out of {n_datasets} requested. "
            f"Processed {len(selected_ids)} candidates."
        )

    if skipped_ids:
        log.warning(
            f"Skipped {len(skipped_ids)} datasets due to issues (missing values or errors): {skipped_ids}"
        )

    return result


def get_series_by_id(
    series_ids: List[str], metadata_path: str, datasets_path: str
) -> Dict[str, Dict]:
    """
    Get metadata and time series data for specific series IDs.

    Args:
        series_ids: List of series IDs to retrieve.
        metadata_path: Path to metadata file.
        datasets_path: Path to parquet file with time series data.

    Returns:
        Dictionary of format {"id": {"metadata": df_row, "timeseries": history}}.

    Raises:
        ValueError: If any series IDs are not found in the metadata.
    """
    # Read and check metadata.
    metadata_df = pd.read_csv(metadata_path)
    result_df = metadata_df[metadata_df["id"].isin(series_ids)]

    found_ids = set(result_df["id"])
    missing_ids = set(series_ids) - found_ids
    if missing_ids:
        raise ValueError(f"Series IDs not found in metadata: {missing_ids}")

    # Read time series data with filtering.
    ts_df = pd.read_parquet(
        datasets_path,
        filters=[("series_id", "in", series_ids)],  # Only read rows we need
    )

    # Build return dictionary.
    result = {}
    skipped_ids = []
    for series_id in series_ids:
        try:
            series_data = (
                ts_df[ts_df["series_id"] == series_id].iloc[0].to_dict()["history"]
            )
            # Check for missing values.
            if any(row[1] == "." for row in series_data):
                log.warning(
                    f"Series {series_id} contains missing values ('.'). Skipping."
                )
                skipped_ids.append(series_id)
                continue

            result[series_id] = {
                "metadata": result_df[result_df["id"] == series_id].iloc[0].to_dict(),
                "timeseries": series_data,
            }
        except (KeyError, IndexError) as e:
            log.warning(f"Error processing series {series_id}: {str(e)}")
            skipped_ids.append(series_id)
            continue

    if not result:
        raise ValueError("No valid data found for any of the specified series")

    if skipped_ids:
        log.warning(
            f"Skipped {len(skipped_ids)} datasets due to issues (missing values or errors): {skipped_ids}"
        )

    return result


def get_dataset_info(fred_data):
    """Create formatted strings for each series with their details."""
    series_details = []
    for series_id, data in fred_data.items():
        length = len(data["timeseries"])
        frequency = data["metadata"].get("frequency", "Unknown")
        title = data["metadata"].get("title", "Unknown")

        detail_str = (
            f"Series ID: {series_id}\n"
            f"  Length: {length} points\n"
            f"  Frequency: {frequency}\n"
            f"  Title: {title}\n"
        )
        series_details.append(detail_str)

    # Join all details with a separator line.
    separator = "-" * 50 + "\n"
    return separator.join(series_details)


def convert_array_to_df(data_array):
    """
    Convert an array of date-value pairs to a DataFrame with 'date' and 'value' columns.
    """
    # Convert array of arrays to list of tuples.
    data_list = [(row[0], float(row[1])) for row in data_array]

    # Create DataFrame.
    df = pd.DataFrame(data_list, columns=["date", "value"])

    # Convert date strings to datetime objects.
    df["date"] = pd.to_datetime(df["date"])

    return df
