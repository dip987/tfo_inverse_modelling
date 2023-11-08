"""
A place to store all the intensity interpolation functions we want try out. Such that it is
available throughout the whole project.
"""
from typing import Callable, Tuple, List, Optional
from pandas import DataFrame
import numpy as np

interpolator_list = []


def _register_func(func: Callable):
    interpolator_list.append(func.__name__)
    return func

@_register_func
def unity_at_zero_interpolation(data: DataFrame, weights: Tuple[float, float], return_alpha: bool = True) -> np.ndarray:
    """Always adds a new point at SDD = 0 with unity intensity. Assumes the rest of the intensty
    starts falling from 1.0. (i.e., none of the other intensity values are above 1.0)

    Args:
        data (DataFrame): A Dataframe containing 'SDD' and 'Intensity' columns
        weights (Tuple[float, float]): left = weight at SDD=0, right = weight at the farthest point
        return_alpha (bool): Should we return the fitting parameters? Defaults to True. Set to False
        to return the interpolated values.

    Returns:
        np.ndarray: fitted parameters
    """
    input_sdd = data["SDD"].to_numpy()
    input_sdd = np.append([0.0], input_sdd)

    input_y = data["Intensity"].to_numpy()
    input_y = np.append([1], input_y)

    X = np.ones((len(input_sdd), 3))  # One column for SDD and one for the bias
    X[:, 0] = input_sdd
    X[:, 1] = np.sqrt(input_sdd)
    X[:, 2] = np.power(input_sdd, 1 / 3)
    Y = np.log10(input_y).reshape(-1, 1)

    # Normalize By Max
    X[:, :] /= np.max(X[:, :], axis=0)

    W = np.diag(_generate_weights_log(weights, (X[0, 0], X[-1, 0]), X[:, 0]))
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y  # Solve
    if return_alpha:
        return beta_hat
    Y_hat = X @ beta_hat
    return Y_hat

@_register_func
def exponenet_3(data: DataFrame, weights: Tuple[float, float], return_alpha: bool = True) -> np.ndarray:
    """
    Fitting with 3 terms - SDD, sqrt(SDD) and 1 (Bias)

    Args:
        data (DataFrame): A Dataframe containing 'SDD' and 'Intensity' columns
        weights (Tuple[float, float]): left = weight at SDD=0, right = weight at the farthest point
        return_alpha (bool): Should we return the fitting parameters? Defaults to True. Set to False
        to return the interpolated values.

    Returns:
        np.ndarray: fitted parameters
    """
    input_sdd = data["SDD"].to_numpy()
    input_y = data["Intensity"].to_numpy()

    X = np.ones((len(input_sdd), 3))  # One column for SDD and one for the bias
    X[:, 0] = input_sdd
    X[:, 1] = np.sqrt(input_sdd)
    Y = np.log10(input_y).reshape(-1, 1)

    # Normalize By Max
    X[:, :] /= np.max(X[:, :], axis=0)

    W = np.diag(_generate_weights_log(weights, (X[0, 0], X[-1, 0]), X[:, 0]))
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y  # Solve
    if return_alpha:
        return beta_hat
    Y_hat = X @ beta_hat
    return Y_hat

@_register_func
def exponenet_4(data: DataFrame, weights: Tuple[float, float], return_alpha: bool = True) -> np.ndarray:
    """
    Fitting with 4 terms - SDD, sqrt(SDD), cubic root(SDD) and 1 (Bias)

    Args:
        data (DataFrame): A Dataframe containing 'SDD' and 'Intensity' columns
        weights (Tuple[float, float]): left = weight at SDD=0, right = weight at the farthest point
        return_alpha (bool): Should we return the fitting parameters? Defaults to True. Set to False
        to return the interpolated values.

    Returns:
        np.ndarray: fitted parameters
    """
    input_sdd = data["SDD"].to_numpy()
    input_y = data["Intensity"].to_numpy()

    X = np.ones((len(input_sdd), 4))  # One column for SDD and one for the bias
    X[:, 0] = input_sdd
    X[:, 1] = np.sqrt(input_sdd)
    X[:, 2] = np.power(input_sdd, 1 / 3)
    Y = np.log10(input_y).reshape(-1, 1)

    # Normalize By Max
    X[:, :] /= np.max(X[:, :], axis=0)

    W = np.diag(_generate_weights_log(weights, (X[0, 0], X[-1, 0]), X[:, 0]))
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y  # Solve
    if return_alpha:
        return beta_hat
    Y_hat = X @ beta_hat
    return Y_hat

@_register_func
def interpolate_exp_cubic_root(data: DataFrame, weights: Tuple[float, float], return_alpha: bool = False) -> np.ndarray:
    """Exponentially interpolate to chunk of data from the same curve to create a denoised
    version of the Intensity. The interpolation uses a weighted version of linear regression.
    (More info here : https://en.wikipedia.org/wiki/Weighted_least_squares)

    Fitting equation used
        log(Intensity) = alpha0 + alpha1 * SDD + alpha2 * sq_root(SDD) + alpha3 * cubic_root(SDD)

    Args:
        data (DataFrame): A chunk of intensity data as Dataframe. The columns should include 'SDD'
        and 'Intensity'
        weights (Tuple[float, float]): Weights used during interpolation. Supply the weight of the
        first and last element as powers of 10. Logarithmically picks the weights for the detectors
        in between.
        return_beta (bool) : Return the fitting parameters instead of the interpolated data.
        Defaults to False

    Returns:
        np.array: returns the interpolated data as a (n x 1) numpy array.
        (Or the fitting parameters, if return_alpha is True)
    """
    # Create the features (X)
    sdd_array = data["SDD"].to_numpy().reshape(-1)
    feature_matrix = np.ones((len(sdd_array), 4))
    feature_matrix[:, 1] = sdd_array
    feature_matrix[:, 2] = np.sqrt(sdd_array)
    feature_matrix[:, 3] = np.power(sdd_array, 1 / 3)

    unfitted_data = np.log(data["Intensity"].to_numpy()).reshape(-1, 1)
    # Define the weight
    diag_1d = _generate_weights_log(weights, (sdd_array[0], sdd_array[1]), data["SDD"].to_numpy())
    w_matrix = np.diag(diag_1d)
    # Solve fitting parameters - alpha_hat
    alpha_hat = (
        np.linalg.inv(feature_matrix.T @ w_matrix @ feature_matrix) @ feature_matrix.T @ w_matrix @ unfitted_data
    )
    if return_alpha:
        return alpha_hat
    y_hat = feature_matrix @ alpha_hat
    return np.exp(y_hat)

@_register_func
def exp_unity_simple(data: DataFrame, weights: Tuple[float, float], return_alpha: bool = False) -> np.ndarray:
    """
    A super simple fitting equation for log intensity of type [InterpolateFuncType], assuming unity at SDD = 0
        log(Intensity) = alpha1 * SDD
    """
    # Create the features (X)
    sdd_array = data["SDD"].to_numpy().reshape(-1)
    feature_matrix = np.ones((len(sdd_array), 1))
    feature_matrix[:, 0] = sdd_array

    unfitted_data = np.log(data["Intensity"].to_numpy()).reshape(-1, 1)
    # Define the weight
    diag_1d = _generate_weights_log(weights, (sdd_array[0], sdd_array[1]), data["SDD"].to_numpy())
    w_matrix = np.diag(diag_1d)
    # Solve fitting parameters - alpha_hat
    alpha_hat = (
        np.linalg.inv(feature_matrix.T @ w_matrix @ feature_matrix) @ feature_matrix.T @ w_matrix @ unfitted_data
    )
    if return_alpha:
        return alpha_hat
    y_hat = feature_matrix @ alpha_hat
    return np.exp(y_hat)

@_register_func
def exp_affine(data: DataFrame, weights: Tuple[float, float], return_alpha: bool = False) -> np.ndarray:
    """
    A simple fitting equation for log intensity of type [InterpolateFuncType], assuming an affine relationship
        log(Intensity) = alpha0 + alpha1 * SDD
    """
    # Create the features (X)
    sdd_array = data["SDD"].to_numpy().reshape(-1)
    feature_matrix = np.ones((len(sdd_array), 2))
    feature_matrix[:, 1] = sdd_array

    unfitted_data = np.log(data["Intensity"].to_numpy()).reshape(-1, 1)
    # Define the weight
    diag_1d = _generate_weights_log(weights, (sdd_array[0], sdd_array[1]), data["SDD"].to_numpy())
    w_matrix = np.diag(diag_1d)
    # Solve fitting parameters - alpha_hat
    alpha_hat = (
        np.linalg.inv(feature_matrix.T @ w_matrix @ feature_matrix) @ feature_matrix.T @ w_matrix @ unfitted_data
    )
    if return_alpha:
        return alpha_hat
    y_hat = feature_matrix @ alpha_hat
    return np.exp(y_hat)

@_register_func
def exp_piecewise_affine(
    data: DataFrame,
    weights: Tuple[float, float],
    return_alpha: bool = False,
    break_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """
    A simple fitting equation for log intensity of type [InterpolateFuncType], assuming a piecewise affine relationship
        log(Intensity) = alpha0 + alpha1 * SDD   (Defined at different SDD values )

    Extra arguments
    break_indices (Optional[List[int]]): SDD indices for the break points of the piecewise linear function. Defaults to
    [0, 5, 12, 20]. Explanation: The first piece lies from 0th to 4th SDD index, the second 5th to 11th index and the
    last(third) piece spans from 12th to the last SDD index.

    Note: For efficiency, this function does not check the condition break_indices length condition.
    """
    # Create default break_indices
    if break_indices is None:
        break_indices = [0, 5, 12, 20]

    # Create the features (X)
    sdd_array = data["SDD"].to_numpy().reshape(-1)
    feature_matrix = np.ones((len(sdd_array), 2))
    feature_matrix[:, 1] = sdd_array

    unfitted_data = np.log(data["Intensity"].to_numpy()).reshape(-1, 1)
    # Define the weight
    diag_1d = _generate_weights_log(weights, (sdd_array[0], sdd_array[1]), data["SDD"].to_numpy())
    w_matrix = np.diag(diag_1d)

    # Create the SDD index slices corresponding to each of the pieces of the piece-wise interpolation
    all_slices = []
    for i in range(len(break_indices) - 1):
        all_slices.append(slice(break_indices[i], break_indices[i + 1]))

    # Solve fitting parameters - alpha_hat, for each of the pieces
    all_alpha_hat = []
    for index_slice in all_slices:
        feature_crop = feature_matrix[index_slice, :]
        w_crop = w_matrix[index_slice, index_slice]
        data_crop = unfitted_data[index_slice, :]
        alpha_hat = np.linalg.inv(feature_crop.T @ w_crop @ feature_crop) @ feature_crop.T @ w_crop @ data_crop
        all_alpha_hat.append(alpha_hat)

    if return_alpha:
        return np.array(all_alpha_hat).flatten()

    # Use fitting parameters to generate interpolated data
    y_hat = np.zeros((len(data), 1))
    for i, index_slice in enumerate(all_slices):
        y_hat[index_slice] = feature_matrix[index_slice, :] @ all_alpha_hat[i]
    return np.exp(y_hat)


def _generate_weights_log(
    weight_range: Tuple[float, float], x_range: Tuple[float, float], detector_x: np.ndarray
) -> List[float]:
    slope = (weight_range[1] - weight_range[0]) / (x_range[1] - x_range[0])
    intercept = weight_range[0] - slope * x_range[0]
    weight_y = [slope * x + intercept for x in detector_x]
    weight_y_log = [10**x for x in weight_y]
    return weight_y_log
