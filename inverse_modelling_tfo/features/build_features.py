"""
Process the simulated data and create proper features that can be passed onto the model
"""
from typing import List, Tuple
from itertools import permutations
from abc import ABC, abstractmethod
from pandas import DataFrame, pivot, merge
from pandas.core.indexes.base import Index
import numpy as np
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params


# TODO: Convert all these functions to objects - should allow better access to related variables!

class FeatureBuilder(ABC):
    """
    Abstract base class to build features from simulation data for training ML models. 
    
    Call build() to generate features as a Pandas DataFrame and use [feature_names] and [labels] to get 
    relevant column names.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names: List[str] = []
        self.labels: List[str] = []
        self.is_ready: bool = False     # Are the feature names ready?

    @abstractmethod
    def build(self) -> DataFrame:
        """
        Build and return the feautes as a DataFrame
        """

class RowCombinationFeatureBuilder(FeatureBuilder):
    """
    Creates new features by combining 2 rows from the given dataset.
    
    The row pairs are chosen such that [fixed_labels] have identical values. While all possible values for the
    [variable_labels] are paired up. (Note: The pairing is a permutation, and NOT a combination). 
    
    The output consists of the [fixed_labels], [var_labels 1] & [var_labels 2] as well as the appened 
    [feature_labels] concatenated for both rows.

    Note: The same row is never selected as both pairs. 
    """
    def __init__(self, data: DataFrame, feature_columns: List[str], fixed_labels: List[str],
                variable_labels: List[str]) -> None:
        super().__init__()
        self.data = data
        self.feature_columns = feature_columns
        self.fixed_labels = fixed_labels
        self.variable_labels = variable_labels
        # Create a possible permutations lookup-table for all possible group lengths
        self.perm_table = {}

    def build(self):
        data_groups = self.data.groupby(self.fixed_labels)
        self._build_perm_table(data_groups.size().unique())
        new_rows = []
        for key, data_group in data_groups:
            index_pairs = self.perm_table[len(data_group)]

            for i, j in index_pairs:
                new_row = np.hstack([data_group.loc[:, self.feature_columns].iloc[i, :], 
                                     data_group.loc[:, self.feature_columns].iloc[j, :],
                                     key, data_group.loc[:, self.variable_labels].iloc[i, :], 
                                     data_group.loc[:, self.variable_labels].iloc[j, :]])
                new_rows.append(new_row)
        new_rows = np.array(new_rows)
        self._create_column_names()
        self.is_ready = True
        return DataFrame(data=new_rows, columns=self.feature_names + self.labels)

    def _build_perm_table(self, available_sizes: np.ndarray):
        for available_size in available_sizes:
            self.perm_table[available_size] = list(permutations(range(available_size), 2))
    
    def _create_column_names(self):
        # Combining two rows into one creates 2x input features - named as x_n
        self.feature_names = [f'x_{n}' for n in range(2 * len(self.feature_columns))]
        # Combininig two rows creates 2 sets of the variable columns - named var 1 and var 2 for each var
        new_variable_columns = [f'{var} 1' for var in self.variable_labels] + \
            [f'{var} 2' for var in self.variable_labels]
        self.labels = self.fixed_labels + new_variable_columns



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
