import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Dict, List, Union


def crps_closed_form(obs, forecasts):
    """
    Computes CRPS using the closed-form expression for an empirical forecast distribution.
    """
    forecasts = np.array(forecasts)

    # mean of the absolute differences
    term1 = np.mean(np.abs(forecasts - obs))

    # average absolute difference between forecasts which is equivalent to
    # the double sum divided by n^2, multiplied by 0.5
    term2 = 0.5 * np.mean(np.abs(forecasts[:, None] - forecasts))
    return term1 - term2


def compute_dataset_metrics(df: pd.DataFrame) -> Dict:
    """
    Computes per-dataset error metrics that can be meaningfully averaged.
    """
    forecast_cols = [col for col in df.columns if col.startswith("forecast_")]
    forecast_matrix = df[forecast_cols].values
    actuals = df["value"].values
    forecast_errors = forecast_matrix - actuals[:, None]

    # Only compute metrics that can be meaningfully averaged
    mae = np.mean(np.abs(forecast_errors))
    rmse = np.sqrt(np.mean(forecast_errors**2))

    # Handle zero values in MAPE calculation
    non_zero_mask = actuals != 0
    if np.any(non_zero_mask):
        mape = (
            np.mean(
                np.abs(forecast_errors[non_zero_mask] / actuals[non_zero_mask, None])
            )
            * 100
        )
    else:
        mape = np.nan  # or some other fallback

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "n_samples": len(actuals),
        # convert to list for JSON serialisation
        "forecasts": forecast_matrix.tolist(),
        "actuals": actuals.tolist(),
    }


def compute_cross_dataset_metrics(
    forecasts: List[np.ndarray], actuals: List[np.ndarray]
) -> Dict:
    """
    Computes metrics that need all datasets together.
    """
    # Combine all forecasts and actuals
    all_forecasts = np.vstack(forecasts)
    all_actuals = np.concatenate(actuals)
    all_errors = all_forecasts - all_actuals[:, None]

    # Rank metrics across all datasets
    ranks = np.apply_along_axis(rankdata, 1, np.abs(all_errors))
    avg_rank = np.mean(ranks)

    # CRPS across all datasets
    crps_values = [
        crps_closed_form(obs, fc) for obs, fc in zip(all_actuals, all_forecasts)
    ]
    avg_crps = np.mean(crps_values)

    # Distribution metrics
    error_percentiles = np.percentile(np.abs(all_errors), [25, 50, 75, 90])

    return {
        "CRPS": float(avg_crps),
        "Average Rank": float(avg_rank),
        "Error P25": float(error_percentiles[0]),
        "Error P50": float(error_percentiles[1]),
        "Error P75": float(error_percentiles[2]),
        "Error P90": float(error_percentiles[3]),
    }


def compute_forecast_metrics(dfs: Union[pd.DataFrame, List[pd.DataFrame]]) -> Dict:
    """
    Computes all metrics, handling both per-dataset and cross-dataset metrics properly.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    # Compute per-dataset metrics
    dataset_metrics = [compute_dataset_metrics(df) for df in dfs]

    # Weight averageable metrics by dataset size
    total_samples = sum(m["n_samples"] for m in dataset_metrics)
    weighted_metrics = {
        "MAE": float(
            sum(m["MAE"] * m["n_samples"] for m in dataset_metrics) / total_samples
        ),
        "RMSE": float(
            np.sqrt(
                sum((m["RMSE"] ** 2 * m["n_samples"]) for m in dataset_metrics)
                / total_samples
            )
        ),
        "MAPE": float(
            sum(m["MAPE"] * m["n_samples"] for m in dataset_metrics) / total_samples
        ),
    }

    # Compute cross-dataset metrics (convert back to np for metrics)
    forecasts = [np.array(m["forecasts"]) for m in dataset_metrics]
    actuals = [np.array(m["actuals"]) for m in dataset_metrics]
    cross_metrics = compute_cross_dataset_metrics(forecasts, actuals)

    # Combine all metrics
    return {**weighted_metrics, **cross_metrics}
