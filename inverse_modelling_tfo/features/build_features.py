"""
Process the simulated data and create proper features that can be passed onto the model
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Literal
from itertools import permutations, combinations
from venv import create
from pandas import DataFrame, pivot, merge
from pandas.core.indexes.base import Index
import numpy as np
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params


class FeatureBuilder(ABC):
    """
    Create features from simulation intensity data pickle file
    """

    @abstractmethod
    def build_feature(self, data: DataFrame) -> DataFrame:
        """
        Build a set of features from the simulation data and return a new dataframe containing the features and
        target labels
        """

    @abstractmethod
    def get_one_word_description(self) -> str:
        """
        A single word description of what this FeatureBuilder does. Useful for making a process flow diagram
        """

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        The column names corresponding to the newly created features
        """

    @abstractmethod
    def get_label_names(self) -> List[str]:
        """
        The column names corresponding to the ground truth labels
        """


class RatioFeatureBuilder(FeatureBuilder):
    """
    Create a new feature by calculating the intensity ratio at each SDD (between 2 wavelengths).

    Depending on if the data is in logarightmic scale, either subtract or just do a ratio. order determines which wave
    length acts as the numerator.
    """

    def __init__(self, intensity_in_log: bool, order: Literal["1", "2"] = "2") -> None:
        super().__init__()
        self.intensity_in_log = intensity_in_log
        self.order = order
        self.feature_names = []
        self.sim_param_columns = []

    def build_feature(self, data: DataFrame) -> DataFrame:
        # Get Intensity for each wavelegth
        wave1 = data[data["Wave Int"] == 1.0].reset_index()["Intensity"]
        wave2 = data[data["Wave Int"] == 2.0].reset_index()["Intensity"]
        # If intensity is in log -> subtract / otherwise divide
        if self.intensity_in_log:
            ratio_feature = (wave2 - wave1) if self.order == "2" else (wave1 - wave2)
        else:
            ratio_feature = (wave2 / wave1) if self.order == "2" else (wave1 / wave2)

        # Create a new df with only a single set of wave int
        data_new = data[data["Wave Int"] == 1.0].drop(columns="Wave Int").reset_index()
        data_new["Ratio"] = ratio_feature

        # Pivot to bring ratio for all SDD into a column single
        self.sim_param_columns = _get_sim_param_columns(data.columns)
        data_new = pivot(data_new, index=self.sim_param_columns, columns=["SDD"], values="Ratio").reset_index()
        # The new ratio columns created have the same name as the SDD value, type of int/float -> our features
        # Python does not seem to like int/float column names -> convert to string first
        self.feature_names = [str(col) for col in data_new.columns if _is_number(col)]
        # Order: The pivot index comes first then the pivot values
        data_new.columns = self.sim_param_columns + self.feature_names
        return data_new

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_label_names(self) -> List[str]:
        return self.sim_param_columns

    def get_one_word_description(self) -> str:
        return "Intensity\nRatio"


class SpatialIntensityFeatureBuilder(FeatureBuilder):
    """
    Gather up the intensity at each SDD for the same simulation parameter combinations and transform it into a single
    row
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_names = []
        self.sim_param_columns = []

    def build_feature(self, data: DataFrame) -> DataFrame:
        self.sim_param_columns = _get_sim_param_columns(data.columns)
        data_new = pivot(
            data, index=self.sim_param_columns, columns=["SDD", "Wave Int"], values="Intensity"
        ).reset_index()
        # Data is going to be multi indexed. Flatten the index
        data_new.columns = [
            "_".join([str(col[0]), str(col[1])]) if col[1] != "" else col[0] for col in data_new.columns
        ]
        # Feature columns are columns that exist in [data_new] but not in sim_param_columns
        self.feature_names = [x for x in data_new.columns if x not in self.sim_param_columns]
        return data_new

    def get_feature_names(self) -> List[str]:
        return self.feature_names

    def get_label_names(self) -> List[str]:
        return self.sim_param_columns

    def get_one_word_description(self) -> str:
        return "Spatial\nIntensity"


def create_row_combos(
    data: DataFrame,
    feature_columns: List[str],
    fixed_labels: List[str],
    variable_labels: List[str],
    perm_or_comb: Literal["perm", "comb"] = "perm",
    combo_count: int = 2,
) -> Tuple[DataFrame, List[str], List[str]]:
    """
    Creates new features by combining [combo_count] number of rows from the given dataset.

    The row groups are chosen such that [fixed_labels] have identical values. While all possible values for the
    [variable_labels] are paired up. (Note: The pairing is a permutation, and NOT a combination).

    The output consists of the [fixed_labels], [var_labels 1] & [var_labels 2] as well as the appened
    [feature_labels] concatenated for both rows.

    change the [perm_or_comb] to 'comb' to get combinations rather than permutations. Use the [comb_count] to change
    the number of rows mixed to generate a single new row

    The output contains the new DataFrame, feature_columns, label_columns
    """
    data_groups = data.groupby(fixed_labels)
    # Create a possible permutations lookup-table for all possible group lengths
    perm_table = _build_perm_table(data_groups.size().unique(), combo_count, perm_or_comb)

    new_rows = []
    for key, data_group in data_groups:
        combo_indices = perm_table[len(data_group)]
        for indices in combo_indices:
            new_row = np.hstack(
                [
                    data_group[feature_columns].iloc[indices].to_numpy().flatten(),
                    key,
                    data_group[variable_labels].iloc[indices].to_numpy().flatten(),
                ]
            )
            new_rows.append(new_row)
    new_rows = np.array(new_rows)
    
    # Create the feature and label names
    feature_names = [f"x_{n}" for n in range(combo_count * len(feature_columns))]
    new_variable_columns = []
    for i in range(combo_count):
        new_variable_columns.append(*[f"{var} {i}" for var in variable_labels])
    labels = fixed_labels + new_variable_columns

    return DataFrame(data=new_rows, columns=feature_names + labels), feature_names, labels


def _build_perm_table(available_sizes: np.ndarray, combo_count: int, perm_or_comb: Literal["perm", "comb"]) -> Dict:
    """Builds all possible pair permutations/combinations of indices for a given set of table lenghts and stores them
    in a Look-up table.

    Args:
        available_sizes (np.ndarray): Available table sizes
        combo_count(int) : How many rows to mix into a single row
        perm_or_comb(Literal['perm', 'comb']) : Whether to use Permutation or Combination

    Returns:
        Dict: Permutation pair look-up Table with the format {table_len: [(ind1, ind2), (ind1, ind3), ...]}
    """
    # Sanity Check
    # TODO: If the table length is smaller than combo_count, throw some sort of error
    mixing_function = combinations if perm_or_comb == "comb" else permutations
    perm_table = {}
    for available_size in available_sizes:
        perm_table[available_size] = np.array(list(mixing_function(range(available_size), combo_count)))
    return perm_table


def create_ror():
    pass
    # TODO:


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


class RowCombinationFeatureBuilder(FeatureBuilder):
    """
    Picks 2 rows and creates a new comination row. In this new row, the labels and features are concatenated
    (horizontally). You can define which features are kept constant and which feature will be permutated/combined to
    generate the pairs
    """

    def __init__(
        self,
        feature_columns: List[str],
        fixed_labels: List[str],
        variable_labels: List[str],
        perm_or_comb: Literal["perm", "comb"] = "perm",
        combo_count: int = 2,
    ) -> None:
        super().__init__()
        self.feature_columns = feature_columns
        self.fixed_labels = fixed_labels
        self.variable_labels = variable_labels
        self.perm_or_comb: Literal["perm", "comb"] = perm_or_comb
        self.combo_count = combo_count
        self._feature = []
        self._label = []

    def build_feature(self, data: DataFrame) -> DataFrame:
        combined_data, self._feature, self._label = create_row_combos(
            data, self.feature_columns, self.fixed_labels, self.variable_labels, self.perm_or_comb, self.combo_count
        )
        return combined_data


class FetalACFeatureBuilder(FeatureBuilder):
    """
    Creates the AC component of the intensity at each SDD for the same simulation parameter combinations by taking the
    difference between intensity.
    """

    def __init__(self, conc_group_column: str, intensity_in_log: bool, perm_or_comb: Literal["perm", "comb"]) -> None:
        """
        Args:
            conc_group_column (str): Column(name) containing a group id(int). Only Data pairs in the same group will be
            used to generate the AC component. For example, Each conc. and a point 5% above/blow it have the can belong
            to let's say group 1. Then these points will be used to create the groups
            intensity_in_log (bool): is the intensity value in log? If in log scale, antilog using 10^x before
            subtracting(to get AC)
            perm_or_comb (Literal['perm', 'comb']): Whether to use Permutation or Combination when generating AC points
            from the data points in the same group
        """
        super().__init__()
        self.conc_group_column = conc_group_column
        self.intensity_in_log = intensity_in_log
        self.perm_or_comb: Literal["perm", "comb"] = perm_or_comb
        self.spatial_intensity_gen = SpatialIntensityFeatureBuilder()
        self._label_names = []
        self._feature_names = []

    def build_feature(self, data: DataFrame) -> DataFrame:
        # Create row combinations for each group
        spatial_data = self.spatial_intensity_gen.build_feature(data)
        intensity_columns = self.spatial_intensity_gen.feature_names  # Sorted Wv1 first then Wv2
        index_columns = self.spatial_intensity_gen.sim_param_columns
        index_columns.remove("Fetal Hb Concentration")
        combination_data, temp_feature_names, self._label_names = create_row_combos(
            spatial_data, intensity_columns, index_columns, ["Fetal Hb Concentration"], self.perm_or_comb, 2
        )

        # Convert those combinations into AC components
        DETECTOR_COUNT = len(intensity_columns) // 2
        self._feature_names = [f"AC_WV1_{i}" for i in range(DETECTOR_COUNT)] + [
            f"AC_WV2_{i}" for i in range(DETECTOR_COUNT)
        ]
        # Calculate the AC component for each detector
        for i in range(DETECTOR_COUNT):
            if self.intensity_in_log:
                term1_wv1 = np.power(10, combination_data[temp_feature_names[i]])
                term1_wv2 = np.power(10, combination_data[temp_feature_names[i + DETECTOR_COUNT]])
                term2_wv1 = np.power(10, combination_data[temp_feature_names[i + DETECTOR_COUNT * 2]])
                term2_wv2 = np.power(10, combination_data[temp_feature_names[i + DETECTOR_COUNT * 3]])
            else:
                term1_wv1 = combination_data[temp_feature_names[i]]
                term1_wv2 = combination_data[temp_feature_names[i + DETECTOR_COUNT]]
                term2_wv1 = combination_data[temp_feature_names[i + DETECTOR_COUNT * 2]]
                term2_wv2 = combination_data[temp_feature_names[i + DETECTOR_COUNT * 3]]
            combination_data[self._feature_names[i]] = term1_wv1 - term2_wv1
            combination_data[self._feature_names[i + DETECTOR_COUNT]] = term1_wv2 - term2_wv2

        # Clean-up intermediate columns
        combination_data.drop(columns=temp_feature_names, inplace=True)
        return combination_data

    def get_one_word_description(self) -> str:
        return "Fetal\nAC"

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def get_label_names(self) -> List[str]:
        return self._label_names


def _get_sim_param_columns(column_names: Index) -> List:
    result = column_names.drop(["SDD", "Intensity", "Wave Int"], errors="ignore")
    return result.to_list()


def _is_number(obj):
    return isinstance(obj, (int, float))
