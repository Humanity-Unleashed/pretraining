import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import rankdata
from typing import Dict, List, Union


def crps_single(obs, forecasts):
    """
    CRPS for a single observation. Uses empirical cumulative distribution.
    """
    sorted_forecasts = np.sort(forecasts)
    n = len(sorted_forecasts)

    def F(x):
        return np.sum(sorted_forecasts <= x) / n

    lower, upper = min(sorted_forecasts[0], obs), max(sorted_forecasts[-1], obs)

    integral, _ = quad(lambda x: (F(x) - (x >= obs)) ** 2, lower, upper)

    return integral


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
    mape = np.mean(np.abs(forecast_errors / actuals[:, None])) * 100

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "n_samples": len(actuals),
        "forecasts": forecast_matrix,
        "actuals": actuals,
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
    crps_values = [crps_single(obs, fc) for obs, fc in zip(all_actuals, all_forecasts)]
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
        "MAE": sum(m["MAE"] * m["n_samples"] for m in dataset_metrics) / total_samples,
        "RMSE": np.sqrt(
            sum((m["RMSE"] ** 2 * m["n_samples"]) for m in dataset_metrics)
            / total_samples
        ),
        "MAPE": sum(m["MAPE"] * m["n_samples"] for m in dataset_metrics)
        / total_samples,
    }

    # Compute cross-dataset metrics
    forecasts = [m["forecasts"] for m in dataset_metrics]
    actuals = [m["actuals"] for m in dataset_metrics]
    cross_metrics = compute_cross_dataset_metrics(forecasts, actuals)

    # Combine all metrics
    return {**weighted_metrics, **cross_metrics}
