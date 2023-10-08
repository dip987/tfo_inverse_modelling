"""
Process the simulated data and create proper features that can be passed onto the model
"""
from typing import List, Tuple
from pandas import DataFrame, pivot, merge
from pandas.core.indexes.base import Index
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params

def create_ratio(data: DataFrame, intensity_in_log: bool) -> DataFrame:
    """Create a Ratio feature from the simulation data
    Ratio is always Wave Int 2 / Wave Int 1

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns. 
        intensity_in_log (bool): Is the intensity value in log? Fale for non-log/regular

    Returns:
        (DataFrame): A new DataFrame with a new Ratio column & (Wave Int, SDD, Intensity) columns
        removed.

    """
    # Create a ratio feature
    wave1 = data[data['Wave Int'] == 1.0].reset_index()['Intensity']
    wave2 = data[data['Wave Int'] == 2.0].reset_index()['Intensity']
    if intensity_in_log:
        ratio_feature = wave2 - wave1
    else:
        ratio_feature = wave2 / wave1

    # Create a new df with only a single set of wave int
    data_new = data[data['Wave Int'] == 1.0].drop(columns='Wave Int').reset_index()
    data_new['Ratio'] = ratio_feature

    # Pivot to bring ratio for all SDD into a column single
    sim_param_columns = _get_sim_param_columns(data.columns)
    data_new = pivot(data_new, index=sim_param_columns, columns=["SDD"], values="Ratio").reset_index()
    # By default, these column names are of int type. Python does not seem to like that
    # Convert to string
    data_new.columns = [str(col) if _is_number(col) else col for col in data_new.columns]
    return data_new


def create_spatial_intensity(data: DataFrame) -> DataFrame:
    """Creates 2 sets of spatial intensity features for each combination of simulation paramter 
    using simulation data. All the features are placed on the same row with column names waveint_sdd
    (Example: 10_1.0)

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns. 

    Returns:
        (DataFrame): A new DataFrame with a new set of spatial intensity column & (Wave Int, SDD,
        Intensity) columns removed.
    """
    sim_param_columns = _get_sim_param_columns(data.columns)
    data_new = pivot(data, index=sim_param_columns, columns=["SDD", "Wave Int"], values="Intensity").reset_index()
    # Data is going to be multi indexed
    data_new.columns = ['_'.join([str(col[0]), str(col[1])]) if col[1] != '' else col[0] for col in data_new.columns]
    return data_new

def create_ratio_and_intensity(data: DataFrame, intensity_in_log: bool) -> DataFrame:
    """Creates spatial intensity & intensity ratio features for each combination of simulation paramter 
    using simulation data. All the features are placed on the same row with column names waveint_sdd
    (Example: 10_1.0) for the spatial intensity and sdd (Example: 1.0) for the ratio.

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns.
        intensity_in_log (bool): Is the intensity value in log? Fale for non-log/regular

    Returns:
        (DataFrame): A new DataFrame with a new set of spatial intensity column & (Wave Int, SDD,
        Intensity) columns removed.
    """
    sim_params = _get_sim_param_columns(data.columns)
    data1 = create_ratio(data, intensity_in_log)
    data2 = create_spatial_intensity(data)
    data = merge(data1, data2, how='inner', on=sim_params)
    return data

def create_curve_fitting_param(data: DataFrame, weights: Tuple[float, float]) -> DataFrame:
    """Creates curve-fitting parameter features for each combination of simulation parameters using simulation data. 
    The features are named as alpha0_1, alpha1_1, ..., alpha0_2, ... in the resultant dataframe

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns.
        weights (Tuple[float, float]): Weights passed on to "get_interpolate_fit_params" function

    Returns:
        DataFrame: A new DataFrame with a new set of spatial intensity column & (Wave Int, SDD,
        Intensity) columns removed and alpha columns added
    """
    sim_params = _get_sim_param_columns(data.columns)
    data1 = get_interpolate_fit_params(data, weights)
    fitting_param_columns = list(filter(lambda X: 'alpha' in X, data1.columns))
    data1 = pivot(data1, index = sim_params, columns=["Wave Int"], values=fitting_param_columns).reset_index()
    # Flatten the multi-index column
    # Afterwards the fitting params should become: alpha0_1, alpha1_1, ..., alpha0_2, ...
    data1.columns = ['_'.join([str(col[0]), str(int(col[1]))]) if col[1] != '' else col[0] for col in data1.columns]
    return data1

def _get_sim_param_columns(column_names: Index) -> List:
    result = column_names.drop(["SDD", "Intensity", "Wave Int"], errors='ignore')
    return result.to_list()

def _is_number(obj):
    return isinstance(obj, (int, float))
