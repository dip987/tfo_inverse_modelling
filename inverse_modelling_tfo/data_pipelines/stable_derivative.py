"""
Interpolate/Smoothen Data using stable derivatives
"""

import numpy as np


def interpolate_pr(data: np.ndarray, derivative_threhold: float = 1e-4, ma_filter_len: int = 3) -> np.ndarray:
    """
    Interpolates the given data using the stable derivative method

    Steps:
    1. Calculate the derivative of the data using the central difference method
    2. 0 out the values which are less than the threshold
    3. Apply a moving average filter to the data (Default length: 3)
    4. Find the first negative gradient and set all the values after that to that negative gradient value

    Args:
        data: np.ndarray: Pulsation Ratio data (against SDD)
        derivative_threhold: float: Threshold value for the derivative. All values less than this will be set to 0.0
        ma_filter_len: int: Length of the moving average filter

    Returns:
        np.ndarray: Interpolated data
    """
    pr_data_derivative = np.gradient(data)
    pr_data_derivative[np.abs(pr_data_derivative) < derivative_threhold] = 0.0
    # Apply a moving average filter
    pr_data_derivative = np.convolve(pr_data_derivative, np.ones(ma_filter_len) / ma_filter_len, mode="same")

    # Find first negative gradient - if exists
    sign_matrix = np.sign(pr_data_derivative)
    left_break_index_matrix = np.where(sign_matrix == -1)[0]
    if len(left_break_index_matrix) != 0:
        left_break_index = left_break_index_matrix[0]
        negative_derivative = pr_data_derivative[left_break_index]
        pr_data_derivative[left_break_index:] = negative_derivative
    interpolated_data = np.cumsum(pr_data_derivative) + data[0]

    return interpolated_data


def interpolate_pr2(data: np.ndarray, derivative_threhold: float = 1e-4, ma_filter_len: int = 3) -> np.ndarray:
    """
    Interpolates the given data using the stable derivative method

    Steps:
    1. Calculate the derivative of the data using the central difference method
    2. 0 out the values which are less than the threshold
    3. Apply a moving average filter to the data (Default length: 3)

    (Difference from interpolate_pr: Ignores step 4, setting all values after the first negative gradient to that
    negative gradient value)

    Args:
        data: np.ndarray: Pulsation Ratio data (against SDD)
        derivative_threhold: float: Threshold value for the derivative. All values less than this will be set to 0.0
        ma_filter_len: int: Length of the moving average filter

    Returns:
        np.ndarray: Interpolated data
    """
    pr_data_derivative = np.gradient(data)
    pr_data_derivative[np.abs(pr_data_derivative) < derivative_threhold] = 0.0
    # Apply a moving average filter
    pr_data_derivative = np.convolve(pr_data_derivative, np.ones(ma_filter_len) / ma_filter_len, mode="same")

    interpolated_data = np.cumsum(pr_data_derivative) + data[0]

    return interpolated_data
