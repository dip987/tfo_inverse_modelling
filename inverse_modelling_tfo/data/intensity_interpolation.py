"""
Functions useful for interpolating spatial intensity values. Mainly useful for denoising and 
determining fitting parameters
"""
from typing import List, Tuple, Callable
import numpy as np
from pandas import DataFrame
from pandas import read_pickle


def interpolate_exp_chunk(data: DataFrame, weights: Tuple[float, float],
                          return_alpha: bool = False) -> np.array:
    """Exponentially interpolate to chunk of data(20 sets of SDD, preferably) to create a denoised 
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
    x = np.ones((len(data), 4))  # One column for SDD and one for the bias
    x[:, 1] = data['SDD'].to_numpy()
    x[:, 2] = np.sqrt(data['SDD'].to_numpy())
    x[:, 3] = np.power(data['SDD'].to_numpy(), 1/3)
    # TODO : Figure out the best interpolation
    Y = np.log(data['Intensity'].to_numpy()).reshape(-1, 1)
    # Define the weight
    W = np.diag(_generate_weights_log(weights, (data['SDD'].to_numpy()[
                0], data['SDD'].to_numpy()[1]), data['SDD'].to_numpy()))
    # W = np.diag(np.logspace(weights[0], weights[1], num=len(Y)))
    alpha_hat = np.linalg.inv(x.T @ W @ x) @ x.T @ W @ Y  # Solve
    if return_alpha:
        return alpha_hat
    else:
        y_hat = x @ alpha_hat
        return np.exp(y_hat)


def interpolate_exp(data: DataFrame, weights: Tuple[float, float] = (1.0, -3),
                    sdd_chunk_size: int = 20) -> DataFrame:
    """Exponentially interpolate to chunk of data(20 sets of SDD, preferably) to create a denoised
    version of the Intensity. The interpolation uses a weighted version of linear regression. 
    (More info here : https://en.wikipedia.org/wiki/Weighted_least_squares)

    Args:
        data (DataFrame): Simulation data loaded in an orderly fashion. (The code expects all 20 
        SDDs to be in a sequence)
        weights (Tuple[float, float], optional): Weights used during interpolation. Supply the 
        weight of the first and last element. Logarithmically picks the middle weights. Defaults to 
        (1.0, -10). (i.e, 10^1 to 10^-10). 
        sdd_chunk_size (int, optional): In case there are a different number of SDD values. 
        Defaults to 20.

    Returns:
        DataFrame: Returns a dataframe with a new column 'Interpolated Intensity' which holds the
        interpolated results 
    """
    interpolated_intensity = None
    for i in range(len(data)//sdd_chunk_size):
        data_chunk = data.iloc[sdd_chunk_size*i: sdd_chunk_size * (i + 1), :]
        y_hat = interpolate_exp_chunk(data_chunk, weights)
        if interpolated_intensity is None:
            interpolated_intensity = y_hat
        else:
            interpolated_intensity = np.vstack([interpolated_intensity, y_hat])
    data['Interpolated Intensity'] = interpolated_intensity
    return data


def get_interpolate_fit_params(data: DataFrame, weights: Tuple[float, float] = (1.0, -2),
                               sdd_chunk_size: int = 20) -> DataFrame:
    """Exponentially interpolate to chunk of data(20 sets of SDD, preferably) to create a
    denoised version of the Intensity. The interpolation uses a weighted version of 
    linear regression.

    This function returns the fitting parameters for the interpolation as alpha0, alpha1, alpha2 
    and alpha3 from the equation below
        log(Intensity) = alpha0 + alpha1 * SDD + alpha2 * sq_root(SDD) + alpha3 * cubic_root(SDD)


    Args:
        data (DataFrame): Simulation data loaded in an orderly fashion. (The code expects all 20 
        SDDs to be in a sequence). It Should have the columns  'SDD' and 'Intensity'
        weights (Tuple[float, float], optional): Weights used during interpolation. Supply the 
        weight of the first and last element. Logarithmically picks the middle weights. 
        Defaults to (1.0, -10), i.e, 10^1 to 10^-10). 
        sdd_chunk_size (int, optional): In case there are a different number of SDD values.
        Defaults to 20.
    Returns:
        DataFrame: Returns a dataframe of fitting parameters for each combination of 
        [wavelength, degrees of freedom] combination.
    """
    model_parameter_columns = data.columns.copy().drop('Intensity').drop('SDD')

    # Do a test run and get the number of fitting parameters
    fit_param_temp = interpolate_exp_chunk(
        data.iloc[0: sdd_chunk_size, :], weights, return_alpha=True)
    fit_param_count = len(fit_param_temp)
    fitting_param_col_names = [f'alpha{x}' for x in range(fit_param_count)]

    fitting_param_table = []

    for i in range(len(data)//sdd_chunk_size):
        data_chunk = data.iloc[sdd_chunk_size*i: sdd_chunk_size * (i + 1), :]
        beta = interpolate_exp_chunk(data_chunk, weights, return_alpha=True)
        fitting_param_table.append(np.hstack(
            [data_chunk.iloc[0][model_parameter_columns].to_numpy(), beta.flatten()]))

    fitting_param_table = DataFrame(data=fitting_param_table, columns=[
                                    *model_parameter_columns, *fitting_param_col_names])
    return fitting_param_table


def get_interpolate_fit_params_custom(data: DataFrame,
                                      custom_fit_func: Callable[[DataFrame, Tuple], np.ndarray],
                                      sdd_chunk_size: int = 20, **kwargs) -> DataFrame:
    """
    Fit a table of simulation data using a custom fitting function. This function will be used on
    [sdd_chunk_size] rows at a time to create a single set of fitting parameters.


    Args:
        data (DataFrame): Simulation data loaded in an orderly fashion. (The code expects all 20 
        SDDs to be in a sequence). It Should have the columns  'SDD' and 'Intensity'
        custom_fit_func (Callable): Function used to generate fitting parameters. It must take 
        atleast 1 inputs, a Dataframe containing two columns, "SDD" and "Intensity". All the keyword
        arguments from the original function will be passed onto this function
        
    Returns:
        DataFrame: Returns a dataframe of fitting parameters for each combination of 
        [wavelength, degrees of freedom] combination. The fitting parameters are named as alpha{n}
        where n ranges from 0 upto (length of fitting params. - 1)
    """
    model_parameter_columns = data.columns.copy().drop('Intensity').drop('SDD')

    # Do a test run and get the number of fitting parameters
    fit_param_temp = custom_fit_func(
        data.iloc[0: sdd_chunk_size, :], **kwargs)
    fit_param_count = len(fit_param_temp)
    fitting_param_col_names = [f'alpha{x}' for x in range(fit_param_count)]

    fitting_param_table = []

    for i in range(len(data)//sdd_chunk_size):
        data_chunk = data.iloc[sdd_chunk_size*i: sdd_chunk_size * (i + 1), :]
        beta = custom_fit_func(data_chunk, **kwargs)
        fitting_param_table.append(np.hstack(
            [data_chunk.iloc[0][model_parameter_columns].to_numpy(), beta.flatten()]))

    fitting_param_table = DataFrame(data=fitting_param_table, columns=[
                                    *model_parameter_columns, *fitting_param_col_names])
    return fitting_param_table


def _generate_weights_log(weight_range: Tuple[float, float], x_range: Tuple[float, float],
                          detector_x: List) -> List:
    slope = (weight_range[1] - weight_range[0])/(x_range[1] - x_range[0])
    intercept = weight_range[0] - slope * x_range[0]
    weight_y = [slope * x + intercept for x in detector_x]
    weight_y_log = [10 ** x for x in weight_y]
    return weight_y_log


if __name__ == "__main__":
    loaded_data = read_pickle(
        r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl')
    # filtered_data = interpolate_exp(loaded_data, weights=[1, -10])
    beta_table = get_interpolate_fit_params(loaded_data, weights=[1, -10])
    print("HALT")
