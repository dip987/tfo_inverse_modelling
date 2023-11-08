"""
Functions useful for interpolating spatial intensity values. Mainly useful for denoising and 
determining fitting parameters
"""
from typing import Tuple, Callable
import numpy as np
from pandas import DataFrame
from .interpolation_function_zoo import interpolate_exp_cubic_root

InterpolatorFuncType = Callable[[DataFrame, Tuple[float, float], bool], np.ndarray]
"""
The custom function MUST accept 3 inputs: The DataFrame itself, A weights tuple and bool [return_alpha]. 
If [return_alpha] is False, the function must return the interpolated column as a numpy array. Otherwise, 
it must return fitting parameters as a numpy array.
"""


def interpolate_exp(
    data: DataFrame,
    weights: Tuple[float, float] = (1.0, -3),
    sdd_chunk_size: int = 20,
    interpolation_function: InterpolatorFuncType = interpolate_exp_cubic_root,
    **interpolation_func_kwargs,
) -> DataFrame:
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
        interpolation_function (InterpolatorFuncType): Which function to use for interpolation. Replace with a custom
        function if required. The custom function must adhere to the InterpolatorFuncType's rules written above
        **interpolation_func_kwargs: You can also pass specific keywords directly onto the interpolation function (On
        top of the 3 already defined in the [InterpolatorFuncType] above)



        The default uses thr fitting equation
            log(Intensity) = alpha0 + alpha1 * SDD + alpha2 * sq_root(SDD) + alpha3 * cubic_root(SDD)

    Returns:
        DataFrame: Returns a dataframe with a new column 'Interpolated Intensity' which holds the
        interpolated results
    """
    interpolated_intensity = None
    # for i in range(len(data)//sdd_chunk_size):
    #     data_chunk = data.iloc[sdd_chunk_size*i: sdd_chunk_size * (i + 1), :]
    #     y_hat = interpolate_exp_chunk(data_chunk, weights)
    #     if interpolated_intensity is None:
    #         interpolated_intensity = y_hat
    #     else:
    #         interpolated_intensity = np.vstack([interpolated_intensity, y_hat])
    for data_chunk in np.array_split(data, len(data) // sdd_chunk_size):
        data_chunk = DataFrame(data_chunk)
        y_hat = interpolation_function(data_chunk, weights, False, **interpolation_func_kwargs)
        if interpolated_intensity is None:
            interpolated_intensity = y_hat
        else:
            interpolated_intensity = np.vstack([interpolated_intensity, y_hat])
    data["Interpolated Intensity"] = interpolated_intensity
    return data


def get_interpolate_fit_params(
    data: DataFrame,
    weights: Tuple[float, float] = (1.0, -2),
    sdd_chunk_size: int = 20,
    custom_fit_func: InterpolatorFuncType = interpolate_exp_cubic_root,
    **interpolation_func_kwargs,
) -> DataFrame:
    """
    Fit a table of simulation data using a custom fitting function. This function will be used on
    [sdd_chunk_size] rows at a time to create a single set of fitting parameters.


    Args:
        data (DataFrame): Simulation data loaded in an orderly fashion. (The code expects all 20
        SDDs to be in a sequence). It Should have the columns  'SDD' and 'Intensity'
        custom_fit_func (InterpolatorFuncType): Function used to generate fitting parameters. The function must adhere
        to the InterpolatorFuncType guidelines

        All the keyword arguments from the original function will be passed onto this function

    Returns:
        DataFrame: Returns a dataframe of fitting parameters for each combination of
        [wavelength, degrees of freedom] combination. The fitting parameters are named as alpha{n}
        where n ranges from 0 upto (length of fitting params. - 1)
    """
    model_parameter_columns = data.columns.copy().drop("Intensity").drop("SDD")

    # Do a test run and get the number of fitting parameters
    fit_param_temp = custom_fit_func(data.iloc[0:sdd_chunk_size, :], weights, True, **interpolation_func_kwargs)
    fit_param_count = len(fit_param_temp)
    fitting_param_col_names = [f"alpha{x}" for x in range(fit_param_count)]

    fitting_param_table = []

    for data_chunk in np.array_split(data, len(data) // sdd_chunk_size):
        data_chunk = DataFrame(data_chunk)
        beta = custom_fit_func(data_chunk, weights, True, **interpolation_func_kwargs)
        fitting_param_table.append(np.hstack([data_chunk.iloc[0][model_parameter_columns].to_numpy(), beta.flatten()]))

    fitting_param_table = DataFrame(
        data=fitting_param_table, columns=[*model_parameter_columns, *fitting_param_col_names]
    )
    return fitting_param_table


def interpolate_exp_transform(
    data: DataFrame, new_sdd: np.ndarray, weights: Tuple[float, float] = (1.0, -3), sdd_chunk_size: int = 20
) -> DataFrame:
    """Exponentially interpolate intensity for a different SDD ([new_sdd]) than given in the original [data]

    This function allows us to convert the current simulation 20 detector setup to our probe 5 detector setup. (Or
    any detector setup defined by their sdd)


    The interpolation uses a weighted version of linear regression.
    (More info here : https://en.wikipedia.org/wiki/Weighted_least_squares)

    Args:
        data (DataFrame): Simulation data loaded in an orderly fashion. (The code expects all 20
        SDDs to be in a sequence)
        new_sdd (List[float]) : SDD list for the new setup
        weights (Tuple[float, float], optional): Weights used during intensity interpolation. Supply the
        weight of the first and last element of the original [data]'s SDD (NOT the [new_sdd]). Logarithmically picks the
        middle weights. Defaults to (1.0, -10). (i.e, 10^1 to 10^-10).
        sdd_chunk_size (int, optional): In case there are a different number of SDD values.
        Defaults to 20.

    Returns:
        DataFrame: Returns a dataframe with a new column 'Interpolated Intensity' which holds the
        interpolated results
    """
    fitting_param_table = get_interpolate_fit_params(data, weights, sdd_chunk_size)
    fitting_param_table = fitting_param_table.reindex(fitting_param_table.index.repeat(len(new_sdd))).reset_index(
        drop=True
    )

    # TODO: Remove the hardcode on determining new interpolated intensity
    new_sdd = np.tile(new_sdd, len(fitting_param_table) // len(new_sdd))
    feature_matrix = np.ones((len(new_sdd), 4))
    feature_matrix[:, 1] = new_sdd
    feature_matrix[:, 2] = np.sqrt(new_sdd)
    feature_matrix[:, 3] = np.power(new_sdd, 1 / 3)

    fitting_param_columns = [f"alpha{x}" for x in range(feature_matrix.shape[1])]
    fitting_params = fitting_param_table.loc[:, fitting_param_columns].to_numpy()
    y = feature_matrix * fitting_params
    y = np.sum(y, axis=1)
    y = np.exp(y)
    fitting_param_table["Intensity"] = y
    fitting_param_table["SDD"] = new_sdd
    fitting_param_table.drop(fitting_param_columns, axis=1, inplace=True)
    return fitting_param_table
