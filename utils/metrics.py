import numpy as np
import xarray as xr


def compute_max_error(pred, target):
    """
    Compute the Maximum Error between two xr.DataArrays.

    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.

    Returns:

        max_error: Maximum Error.
    """
    error = np.abs(pred - target)
    max_error = error.max()
    return max_error


def compute_99999_error(pred, target):
    """
    Compute the 0.99999 Quantile Error between two xr.DataArrays.

    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.

    Returns:
        quantile_error: Quantile Error.
    """
    error = np.abs(pred - target)
    quantile_error = np.quantile(error, 0.99999)
    return quantile_error


def compute_rmse(pred, target, mean_dims=None):
    """
    Compute the Root Mean Squared Error (RMSE) between two xr.DataArrays.

    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.
        mean_dims: Dimensions over which to average the squared error.

    Returns:
        rmse: Root Mean Squared Error.
    """
    error = pred - target
    squared_error = error**2
    if mean_dims is not None:
        squared_error = squared_error.mean(dim=mean_dims)
    rmse = np.sqrt(squared_error.mean())
    return rmse


def compute_mae(pred, target, mean_dims=None):
    """
    Compute the Mean Absolute Error (MAE) between two xr.DataArrays.
    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.
        mean_dims: Dimensions over which to average the absolute error.
    Returns:
        mae: Mean Absolute Error.
    """
    error = pred - target
    absolute_error = np.abs(error)
    if mean_dims is not None:
        absolute_error = absolute_error.mean(dim=mean_dims)
    mae = absolute_error.mean()
    return mae


def compute_weighted_rmse(pred, target, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting between two xr.DataArrays.
    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = pred - target
    weights_lat = np.cos(np.deg2rad(error.latitude))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error) ** 2 * weights_lat).mean(mean_dims))
    return rmse


def compute_weighted_mae(pred, target, mean_dims=xr.ALL_DIMS):
    """
    Compute the MAE with latitude weighting between two xr.DataArrays.
    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = pred - target
    weights_lat = np.cos(np.deg2rad(error.latitude))
    weights_lat /= weights_lat.mean()
    mae = (np.abs(error) * weights_lat).mean(mean_dims)
    return mae


def compute_psnr(pred, target):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two xr.DataArrays.

    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.

    Returns:
        psnr: Peak Signal-to-Noise Ratio.
    """
    mse = np.mean((pred - target) ** 2)
    max_range = np.max(target) - np.min(target)
    psnr = 20 * np.log10(max_range) - 10 * np.log10(mse)
    return psnr


def compute_evaluation_metrics(pred, target, prefix="test"):
    """
    Compute evaluation metrics between predicted and target xr.DataArrays.

    Args:
        pred (xr.DataArray): Forecast.
        target (xr.DataArray): Truth.

    Returns:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    metrics = {}

    # Compute max_error
    max_error = compute_max_error(pred, target)
    metrics[f"{prefix}_max_error"] = max_error.item()

    # Compute 0.99999 error
    quantile_error = compute_99999_error(pred, target)
    metrics[f"{prefix}_0.99999_quantile_error"] = quantile_error.item()

    # Compute rmse
    rmse = compute_rmse(pred, target)
    metrics[f"{prefix}_rmse"] = rmse.item()

    # Compute mae
    mae = compute_mae(pred, target)
    metrics[f"{prefix}_mae"] = mae.item()

    # Compute weighted rmse
    rmse = compute_weighted_rmse(pred, target)
    metrics[f"{prefix}_weighted_rmse"] = rmse.item()

    # Compute weighted mae
    mae = compute_weighted_mae(pred, target)
    metrics[f"{prefix}_weighted_mae"] = mae.item()

    # Compute psnr
    psnr = compute_psnr(pred, target)
    metrics[f"{prefix}_psnr"] = psnr.item()

    return metrics


def compute_mean_metrics(dict_list):
    """
    Compute mean values of metrics for each pressure level.

    Args:
        dict_list (list): List of dictionaries with metric scores for each pressure level.

    Returns:
        mean_dict (dict): Dictionary containing mean evaluation metrics.
    """
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[f"mean_{key}"] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict
