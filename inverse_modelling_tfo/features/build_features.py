"""
Process the simulated data and create proper features that can be passed onto the model
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Literal
from itertools import permutations, combinations, product
from pandas import DataFrame, pivot, merge
from pandas.core.indexes.base import Index
import numpy as np
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params

WAVE_INT_COUNT = 2


class FeatureBuilder(ABC):
    """
    Create features from using simulation intensity data(pickle file)
    The data should always have the following columns:
    - SDD
    - Intensity at each SDD(as separate columns)
    - Wave Int
    - Tissue Model Parameters (e.g. Fetal Hb Concentration, Maternal Saturation, etc.)
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
        self._label = []

    def build_feature(self, data: DataFrame) -> DataFrame:
        data_groups = data.groupby(self.fixed_labels)
        # Create a possible permutations lookup-table for all possible group lengths and cache them
        perm_table = _build_perm_table(data_groups.size().unique(), self.combo_count, self.perm_or_comb)

        combination_table = []
        for key, data_group in data_groups:
            combo_indices = perm_table[len(data_group)]
            for indices in combo_indices:
                new_row = np.hstack(
                    [
                        data_group[self.feature_columns].iloc[indices].to_numpy().flatten(),
                        key,
                        data_group[self.variable_labels].iloc[indices].to_numpy().flatten(),
                    ]
                )
                # Column ordering: feature_columns, fixed_labels, variable_labels
                combination_table.append(new_row)
        combination_table = np.array(combination_table)
        combination_df = DataFrame(data=combination_table, columns=self.get_feature_names() + self.get_label_names())
        return combination_df

    def get_one_word_description(self) -> str:
        return "Row\nCombination"

    def get_feature_names(self) -> List[str]:
        return [f"x_{n}" for n in range(self.combo_count * len(self.feature_columns))]

    def get_label_names(self) -> List[str]:
        new_variable_columns = []
        for i in range(self.combo_count):
            new_variable_columns.append(*[f"{var} {i}" for var in self.variable_labels])

        # DO NOT change this ordering
        return self.fixed_labels + new_variable_columns


class FetalACFeatureBuilder(FeatureBuilder):
    """
    Creates the AC component of the intensity at each SDD for the same simulation parameter combinations by taking the
    difference between intensity.
    """

    def __init__(
        self,
        conc_group_column: str,
        perm_or_comb: Literal["perm", "comb"],
        mode: Literal["-", "/"],
        spatial_intensity_columns: List[str],
        labels: List[str],
    ) -> None:
        """
        Args:
            conc_group_column (str): Column(name) containing a group id(int). Only Data pairs in the same group will be
            used to generate the AC component. For example, Each conc. and a point 5% above/blow it have the can belong
            to let's say group 1. Then these points will be used to create the groups
            mode(Literal["-", "/"]): Whetter to use subtraction or division to calculate the AC component
            perm_or_comb (Literal['perm', 'comb']): Whether to use Permutation or Combination when generating AC points
            from the data points in the same group
            spatial_intensity_columns (List[str]): List of column names containing the spatial intensity values
            labels (List[str]): List of column names containing the simulation parameters (Stays untouched)
        """
        self.conc_group_column = conc_group_column
        self.mode = mode
        self.perm_or_comb: Literal["perm", "comb"] = perm_or_comb
        self._label_names = labels
        self.intensity_columns = spatial_intensity_columns  # Sorted Wv1 first then Wv2
        fixed_labels = self._create_combination_index_columns()
        self.combination_feature_builder = RowCombinationFeatureBuilder(
            self.intensity_columns, fixed_labels, ["Fetal Hb Concentration"], self.perm_or_comb, 2
        )

    def build_feature(self, data: DataFrame) -> DataFrame:
        # Create row combinations for each group
        combination_data = self.combination_feature_builder.build_feature(data)
        combination_feature_names = self.combination_feature_builder.get_feature_names()

        # Convert those combinations into AC components
        detector_count = self._get_detector_count()
        operation = np.subtract if self.mode == "-" else np.divide

        # Calculate the AC component for each detector # Assuming 2 Wave Ints
        _feature_names = self.get_feature_names()
        for i in range(detector_count):
            term1_wv1 = combination_data[combination_feature_names[i]]
            term1_wv2 = combination_data[combination_feature_names[i + detector_count]]
            term2_wv1 = combination_data[combination_feature_names[i + detector_count * 2]]
            term2_wv2 = combination_data[combination_feature_names[i + detector_count * 3]]

            combination_data[_feature_names[i]] = operation(term1_wv1, term2_wv1)
            combination_data[_feature_names[i + detector_count]] = operation(term1_wv2, term2_wv2)

        # Clean-up intermediate columns
        combination_data.drop(columns=combination_feature_names, inplace=True)
        return combination_data

    def get_one_word_description(self) -> str:
        return "Fetal\nAC"

    def get_feature_names(self) -> List[str]:
        detector_count = self._get_detector_count()
        feature_name_appender = "Sub" if self.mode == "-" else "Div"
        id_pairs = product(range(1, WAVE_INT_COUNT + 1), range(detector_count))
        return [f"{feature_name_appender}AC_WV{wave_int}_{detector_index}" for wave_int, detector_index in id_pairs]

    def get_label_names(self) -> List[str]:
        return self.combination_feature_builder.get_label_names()

    def _create_combination_index_columns(self) -> List[str]:
        index_columns = self._label_names.copy()
        index_columns.remove("Fetal Hb Concentration")
        if self.conc_group_column not in index_columns:
            index_columns.append(self.conc_group_column)
        return index_columns

    def _get_detector_count(self) -> int:
        return len(self.intensity_columns) // WAVE_INT_COUNT


class FetalACbyDCFeatureBuilder(FeatureBuilder):
    """
    Creates an AC by DC. The AC is calculated by taking the difference between intensity for two rows with
    same TMP and different Fetal Hb Concentration. Considers [conc_group_column] during picking two rows. Only rows with
    the same id on this column will be paired up(With the rest of the TMPs other than Fetal Hb Conc) being same. The DC
    is taken as the smaller of the two intensities. The AC by DC is then calculated by dividing the AC by the DC.
    Note: Does not perform any abs() operation on the AC or DC. So the result can be negative!
    """

    def __init__(
        self,
        conc_group_column: str,
        perm_or_comb: Literal["perm", "comb"],
        spatial_intensity_columns: List[str],
        labels: List[str],
        dc_mode: Literal["max", "min"] = "min",
    ) -> None:
        """
        Args:
            conc_group_column (str): Column(name) containing a group id(int). Only Data pairs in the same group will be
            used to generate the AC component. For example, Each conc. and a point 5% above/blow it have the can belong
            to let's say group 1. Then these points will be used to create the groups
            perm_or_comb (Literal['perm', 'comb']): Whether to use Permutation or Combination when generating point
            pairs from the data points in the same group
            dc_mode: Literal["max", "min"]: Defaults to "min". Which Intensity goes in the denominator. "min" for the
            smaller of the two intensities and "max" for the larger of the two intensities
        """
        super().__init__()
        self.conc_group_column = conc_group_column
        self.perm_or_comb: Literal["perm", "comb"] = perm_or_comb
        self.dc_mode = dc_mode
        self._feature_names = []
        self._label_names = labels
        self.intensity_columns = spatial_intensity_columns  # Sorted Wv1 first then Wv2
        fixed_labels = self._create_combination_index_columns()
        self.combination_feature_builder = RowCombinationFeatureBuilder(
            self.intensity_columns, fixed_labels, ["Fetal Hb Concentration"], self.perm_or_comb, 2
        )

    def build_feature(self, data: DataFrame) -> DataFrame:
        # Create row combinations for each group
        combination_data = self.combination_feature_builder.build_feature(data)
        combination_feature_names = self.combination_feature_builder.get_feature_names()
        detector_count = self._get_detector_count()

        # Calculate the AC component for each detector # Assuming 2 Wave Ints
        _feature_names = self.get_feature_names()
        for i in range(detector_count):
            term1_wv1 = combination_data[combination_feature_names[i]]
            term1_wv2 = combination_data[combination_feature_names[i + detector_count]]
            term2_wv1 = combination_data[combination_feature_names[i + detector_count * 2]]
            term2_wv2 = combination_data[combination_feature_names[i + detector_count * 3]]

            denominator_func = np.minimum if self.dc_mode == "min" else np.maximum
            numerator_wv1 = np.subtract(term1_wv1, term2_wv1)
            denominator_wv1 = denominator_func(term1_wv1, term2_wv1)

            numerator_wv2 = np.subtract(term1_wv2, term2_wv2)
            denominator_wv2 = denominator_func(term1_wv2, term2_wv2)

            # DO NOT Change this ordering
            combination_data[_feature_names[i]] = numerator_wv1 / denominator_wv1
            combination_data[_feature_names[i + detector_count]] = numerator_wv2 / denominator_wv2

        # Clean-up intermediate columns
        combination_data.drop(columns=combination_feature_names, inplace=True)
        return combination_data

    def get_one_word_description(self) -> str:
        return "Fetal\nACbyDC"

    def get_feature_names(self) -> List[str]:
        detector_count = self._get_detector_count()
        feature_name_appender = "MIN" if self.dc_mode == "min" else "MAX"
        # DO NOT Change this ordering
        id_pairs = product(range(1, WAVE_INT_COUNT + 1), range(detector_count))
        return [
            f"{feature_name_appender}_ACbyDC_WV{wave_int}_{detector_index}" for wave_int, detector_index in id_pairs
        ]

    def get_label_names(self) -> List[str]:
        return self.combination_feature_builder.get_label_names()

    def _create_combination_index_columns(self) -> List[str]:
        index_columns = self._label_names.copy()
        index_columns.remove("Fetal Hb Concentration")
        if self.conc_group_column not in index_columns:
            index_columns.append(self.conc_group_column)
        return index_columns

    def _get_detector_count(self) -> int:
        return len(self.intensity_columns) // WAVE_INT_COUNT


class TwoColumnOperationFeatureBuilder(FeatureBuilder):
    """
    Create a new feature by applying an operation( +, -, /, *) on two sets of feature columns (one to one mapped between
    term1 and term2)
    """

    operator_map = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}

    def __init__(
        self,
        term1_cols: List[str],
        term2_cols: List[str],
        operator: Literal["+", "-", "*", "/"],
        keep_original: bool,
        feature_columns: List[str],
        labels: List[str],
    ) -> None:
        # Check if term1 and term2 rows have the same length
        if len(term1_cols) != len(term2_cols):
            raise ValueError("Numerator and denominator rows must have the same length")
        self.term1_cols = term1_cols
        self.term2_cols = term2_cols
        self.keep_original = keep_original
        self._labels = labels
        self.old_features = feature_columns
        self.operator_func = TwoColumnOperationFeatureBuilder.operator_map[operator]
        self.operator = operator
        self.new_features = self._generate_new_feature_names()

    def build_feature(self, data: DataFrame) -> DataFrame:
        # Do not modiofy the original data
        data = data.copy()
        # Check if term1 and term2 rows exist in the data
        if not all([row in data.columns for row in self.term1_cols]):
            raise ValueError("Numerator rows do not exist in the data")
        if not all([row in data.columns for row in self.term2_cols]):
            raise ValueError("Denominator rows do not exist in the data")

        # Generate Features
        new_features = self._generate_new_feature_names()
        for term1, term2, new_feature_name in zip(self.term1_cols, self.term2_cols, new_features):
            data[new_feature_name] = self.operator_func(data[term1], data[term2])

        # Cleanup
        if not self.keep_original:
            self._delete_old_features(data)

        return data

    def _delete_old_features(self, data: DataFrame):
        for numerator, denominator in zip(self.term1_cols, self.term2_cols):
            data.drop(columns=[numerator, denominator], inplace=True)

    def get_feature_names(self) -> List[str]:
        if self.keep_original:
            return self.old_features + self.new_features
        old_features_remaining = [feature for feature in self.old_features if (feature not in self.term1_cols) and (feature not in self.term2_cols)]
        return old_features_remaining + self.new_features

    def get_label_names(self) -> List[str]:
        return self._labels

    def get_one_word_description(self) -> str:
        return "Division\nFeature"

    def _generate_new_feature_names(self) -> List[str]:
        feature_names = [f"{term1}_{self.operator}_{term2}" for term1, term2 in zip(self.term1_cols, self.term2_cols)]
        return feature_names


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
