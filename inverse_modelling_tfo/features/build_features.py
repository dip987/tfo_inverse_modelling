"""
Process the simulated data and create proper features that can be passed onto the model
"""
from typing import List, Tuple, Dict
from itertools import permutations
from pandas import DataFrame, pivot, merge
from pandas.core.indexes.base import Index
import numpy as np
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params


def create_row_combos(
    data: DataFrame, feature_columns: List[str], fixed_labels: List[str], variable_labels: List[str]
) -> Tuple[DataFrame, List[str], List[str]]:
    """
    Creates new features by combining 2 rows from the given dataset.

    The row pairs are chosen such that [fixed_labels] have identical values. While all possible values for the
    [variable_labels] are paired up. (Note: The pairing is a permutation, and NOT a combination).

    The output consists of the [fixed_labels], [var_labels 1] & [var_labels 2] as well as the appened
    [feature_labels] concatenated for both rows.

    The output contains the new DataFrame, feature_columns, label_columns
    """
    data_groups = data.groupby(fixed_labels)
    # Create a possible permutations lookup-table for all possible group lengths
    perm_table = _build_perm_table(data_groups.size().unique())

    new_rows = []
    for key, data_group in data_groups:
        index_pairs = perm_table[len(data_group)]

        for i, j in index_pairs:
            new_row = np.hstack(
                [
                    data_group.loc[:, feature_columns].iloc[i, :],
                    data_group.loc[:, feature_columns].iloc[j, :],
                    key,
                    data_group.loc[:, variable_labels].iloc[i, :],
                    data_group.loc[:, variable_labels].iloc[j, :],
                ]
            )
            new_rows.append(new_row)
    new_rows = np.array(new_rows)

    # Create the feature and label names
    feature_names = [f"x_{n}" for n in range(2 * len(feature_columns))]
    new_variable_columns = [f"{var} 1" for var in variable_labels] + [f"{var} 2" for var in variable_labels]
    labels = fixed_labels + new_variable_columns

    return DataFrame(data=new_rows, columns=feature_names + labels), feature_names, labels


def _build_perm_table(available_sizes: np.ndarray) -> Dict:
    """Builds all possible pair permutations of indices for a given set of table lenghts and stores them in a Look-up
    table.

    Args:
        available_sizes (np.ndarray): Available table sizes

    Returns:
        Dict: Permutation pair look-up Table with the format {table_len: [(ind1, ind2), (ind1, ind3), ...]}
    """
    perm_table = {}
    for available_size in available_sizes:
        perm_table[available_size] = list(permutations(range(available_size), 2))
    return perm_table


def create_ratio(data: DataFrame, intensity_in_log: bool) -> Tuple[DataFrame, List[str], List[str]]:
    """Create a Ratio feature from the simulation data
    Ratio is always Wave Int 2 / Wave Int 1

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns.
        intensity_in_log (bool): Is the intensity value in log? Fale for non-log/regular

    Returns:
        A new DataFrame with a new Ratio column & (Wave Int, SDD, Intensity) columns removed, Feature Names, Labels

    """
    # Create a ratio feature
    wave1 = data[data["Wave Int"] == 1.0].reset_index()["Intensity"]
    wave2 = data[data["Wave Int"] == 2.0].reset_index()["Intensity"]
    if intensity_in_log:
        ratio_feature = wave2 - wave1
    else:
        ratio_feature = wave2 / wave1

    # Create a new df with only a single set of wave int
    data_new = data[data["Wave Int"] == 1.0].drop(columns="Wave Int").reset_index()
    data_new["Ratio"] = ratio_feature

    # Pivot to bring ratio for all SDD into a column single
    sim_param_columns = _get_sim_param_columns(data.columns)
    data_new = pivot(data_new, index=sim_param_columns, columns=["SDD"], values="Ratio").reset_index()
    # The new ratio columns created have the same name as the SDD value, type of int/float -> our features
    # Python does not seem to like int/float column names -> convert to string first
    feature_names = [str(col) for col in data_new.columns if _is_number(col)]
    # Order: The pivot index comes first then the pivot values
    data_new.columns = sim_param_columns + feature_names
    return data_new, feature_names, sim_param_columns


def create_spatial_intensity(data: DataFrame) -> Tuple[DataFrame, List[str], List[str]]:
    """Creates 2 sets of spatial intensity features for each combination of simulation paramter
    using simulation data. All the features are placed on the same row with column names waveint_sdd
    (Example: 10_1.0)

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns.

    Returns:
        A new DataFrame with a new set of spatial intensity column & (Wave Int, SDD, Intensity) columns removed,
        Feature Names, Labels
    """
    sim_param_columns = _get_sim_param_columns(data.columns)
    data_new = pivot(data, index=sim_param_columns, columns=["SDD", "Wave Int"], values="Intensity").reset_index()
    # Data is going to be multi indexed. Flatten the index
    data_new.columns = ["_".join([str(col[0]), str(col[1])]) if col[1] != "" else col[0] for col in data_new.columns]
    # Feature columns are columns that exist in [data_new] but not in sim_param_columns
    feature_columns = [x for x in data_new.columns if x not in sim_param_columns]
    return data_new, feature_columns, sim_param_columns


def create_ratio_and_intensity(data: DataFrame, intensity_in_log: bool) -> Tuple[DataFrame, List[str], List[str]]:
    """Creates spatial intensity & intensity ratio features for each combination of simulation paramter
    using simulation data. All the features are placed on the same row with column names waveint_sdd
    (Example: 10_1.0) for the spatial intensity and sdd (Example: 1.0) for the ratio.

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns.
        intensity_in_log (bool): Is the intensity value in log? Fale for non-log/regular

    Returns:
        A new DataFrame with a new set of spatial intensity column & (Wave Int, SDD) columns removed, Feature Names,
        Labels
    """
    sim_params = _get_sim_param_columns(data.columns)
    data1, features1, _ = create_ratio(data, intensity_in_log)
    data2, features2, _ = create_spatial_intensity(data)
    data = merge(data1, data2, how="inner", on=sim_params)
    return data, features1 + features2, sim_params


def create_curve_fitting_param(data: DataFrame, weights: Tuple[float, float]) -> Tuple[DataFrame, List[str], List[str]]:
    """Creates curve-fitting parameter features for each combination of simulation parameters using simulation data.
    The features are named as alpha0_1, alpha1_1, ..., alpha0_2, ... in the resultant dataframe

    Args:
        data (DataFrame): Simulation data with "Intensity", "SDD" & "Wave Int" Columns.
        weights (Tuple[float, float]): Weights passed on to "get_interpolate_fit_params" function

    Returns:
        A new DataFrame with a new set of spatial intensity column & (Wave Int, SDD, Intensity) columns removed and
        alpha columns added, Feature Names, Labels
    """
    sim_params = _get_sim_param_columns(data.columns)
    data1 = get_interpolate_fit_params(data, weights)
    fitting_param_columns = list(filter(lambda X: "alpha" in X, data1.columns))
    data1 = pivot(data1, index=sim_params, columns=["Wave Int"], values=fitting_param_columns).reset_index()
    # Flatten the multi-index column
    # Afterwards the fitting params should become: alpha0_1, alpha1_1, ..., alpha0_2, ...
    data1.columns = ["_".join([str(col[0]), str(int(col[1]))]) if col[1] != "" else col[0] for col in data1.columns]
    # The new feature columns  (fitting param columns) exist in the [data1.columns] but not in sim_params
    feature_columns = [x for x in data1.columns if x not in sim_params]
    return data1, feature_columns, sim_params


def _get_sim_param_columns(column_names: Index) -> List:
    result = column_names.drop(["SDD", "Intensity", "Wave Int"], errors="ignore")
    return result.to_list()


def _is_number(obj):
    return isinstance(obj, (int, float))
