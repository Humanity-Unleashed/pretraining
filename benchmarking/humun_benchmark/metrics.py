import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.integrate import quad


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


def compute_forecast_metrics(df: pd.DataFrame):
    """
    Computes multiple error metrics for a set of forecasts.
    Assumes 'value' is the true value and 'forecast_1', 'forecast_2', ... are the predictions.

    NOTE: error metrics (mae, rmse, mape) are aggregated by taking the mean of each metric.
        i.e. mae = 1/n * sum(n_maes)
    """

    forecast_cols = [col for col in df.columns if col.startswith("forecast_")]
    forecast_matrix = df[forecast_cols].values
    actuals = df["value"].values

    forecast_errors = (
        forecast_matrix - actuals[:, None]
    )  # Shape (n_samples, n_forecasts)

    # aggregated metrics
    mae = np.mean(np.abs(forecast_errors))
    rmse = np.sqrt(np.mean(forecast_errors**2))
    mape = np.mean(np.abs(forecast_errors / actuals[:, None])) * 100

    # compute CRPS across all time steps
    crps_values = [
        crps_single(obs, forecasts) for obs, forecasts in zip(actuals, forecast_matrix)
    ]
    avg_crps = np.mean(crps_values)

    ranks = np.apply_along_axis(rankdata, 1, np.abs(forecast_errors))
    avg_rank = np.mean(ranks)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "CRPS": float(avg_crps),
        "Average Rank": float(avg_rank),
    }
