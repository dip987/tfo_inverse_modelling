"""
A place to store all the intensity interpolation functions we want try out. Such that it is
available throughout the whole project.
"""
from typing import Tuple
from pandas import DataFrame
import numpy as np
from inverse_modelling_tfo.data.intensity_interpolation import _generate_weights_log


def unity_at_zero_interpolation(data: DataFrame, weights: Tuple[float, float],
                                return_alpha: bool = True) -> np.ndarray:
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
    input_sdd = data['SDD'].to_numpy()
    input_sdd = np.append([0.0], input_sdd)

    input_y = data['Intensity'].to_numpy()
    input_y = np.append([1], input_y)

    X = np.ones((len(input_sdd), 3))  # One column for SDD and one for the bias
    X[:, 0] = input_sdd
    X[:, 1] = np.sqrt(input_sdd)
    X[:, 2] = np.power(input_sdd, 1/3)
    Y = np.log10(input_y).reshape(-1, 1)

    # Normalize By Max
    X[:, :] /= np.max(X[:, :], axis=0)

    W = np.diag(_generate_weights_log(weights, (X[0, 0], X[-1, 0]), X[:, 0]))
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y  # Solve
    if return_alpha:
        return beta_hat
    Y_hat = X @ beta_hat
    return Y_hat


def exponenet_3(data: DataFrame, weights: Tuple[float, float],
                return_alpha: bool = True) -> np.ndarray:
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
    input_sdd = data['SDD'].to_numpy()
    input_y = data['Intensity'].to_numpy()

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


def exponenet_4(data: DataFrame, weights: Tuple[float, float],
                return_alpha: bool = True) -> np.ndarray:
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
    input_sdd = data['SDD'].to_numpy()
    input_y = data['Intensity'].to_numpy()

    X = np.ones((len(input_sdd), 4))  # One column for SDD and one for the bias
    X[:, 0] = input_sdd
    X[:, 1] = np.sqrt(input_sdd)
    X[:, 2] = np.power(input_sdd, 1/3)
    Y = np.log10(input_y).reshape(-1, 1)

    # Normalize By Max
    X[:, :] /= np.max(X[:, :], axis=0)

    W = np.diag(_generate_weights_log(weights, (X[0, 0], X[-1, 0]), X[:, 0]))
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y  # Solve
    if return_alpha:
        return beta_hat
    Y_hat = X @ beta_hat
    return Y_hat
