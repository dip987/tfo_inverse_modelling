"""
Process the simulated data and create proper features that can be passed onto the model
"""

from abc import abstractmethod
from typing import List, Literal
from itertools import product
from pandas import DataFrame
import numpy as np
from inverse_modelling_tfo.features.data_transformations import DataTransformation
from .build_features_helpers import _build_perm_table, TypePairing


# TODO: There is a naming inconsistency in the code. Some places have the detector index, some use its SDD. Fix this
WAVE_INT_COUNT = 2


class FeatureBuilder(DataTransformation):
    """
    Create features from using simulation intensity data(pickle file)
    The data should always have the following columns:
    - SDD
    - Intensity at each SDD(as separate columns)
    - Wave Int
    - Tissue Model Parameters (e.g. Fetal Hb Concentration, Maternal Saturation, etc.)
    """

    def __init__(self):
        super().__init__()
        self.chain = None

    @abstractmethod
    def build_feature(self, data: DataFrame) -> DataFrame:
        """
        Build a set of features from the simulation data and return a new dataframe containing the features and
        target labels
        """

    @classmethod
    @abstractmethod
    def from_chain(cls, chain: DataTransformation, *args, **kwargs) -> "FeatureBuilder":
        """
        Create a FeatureBuilder which can be added on top of another DataTransformation as a chain. This passes the
        chain's labels and features to the new FeatureBuilder object automatically. Applying the new FeatureBuilder
        will also apply all the transformations in the chain.
        Example:
        ----------
        fb1 = FeatureBuilder1(..)
        labels = fb1.get_label_names()  # DataTransformation method
        features = fb1.get_feature_names()  # DataTransformation method
        fb2 = FeatureBuilder2(labels, features, ..)
        data = fb1(data)
        data = fb2(data)

        This is analogous to:
        fb1 = FeatureBuilder1(..)
        fb2 = FeatureBuilder2.from_chain(fb1, ..)
        data = fb2(data)


        The chain's transform/build_feature should be called during the class's own build_feature method (Preferably
        at the very first line).
        """

    def transform(self, data: DataFrame) -> DataFrame:
        return self.build_feature(data)

    def _apply_chain(self, data: DataFrame) -> DataFrame:
        """
        Only apply the chain if it is FeatureBuilder object. Otherwise, return the data as is. Common class method
        accessible to all subclasses. Each class implementation has the freedom to choose how they want to apply the
        chain
        """
        if self.chain is None:
            return data
        elif isinstance(self.chain, FeatureBuilder):
            # This returns a new copy of the data (OG stays unaltered)
            return self.chain.transform(data)
        elif isinstance(self.chain, DataTransformation):
            return data
        else:
            raise ValueError("This object's 'chain' atrribute must be a None or a DataTransformation object")


class RowCombinationFeatureBuilder(FeatureBuilder):
    """
    Picks 2 rows and creates a new comination row. In this new row, the labels and features are concatenated
    (horizontally). You can define which features are kept constant and which feature will be permutated/combined to
    generate the pairs
    Ordering of the columns in the new row:
    - feature_columns
    - fixed_labels
    - variable_labels 1
    - variable_labels 2

    """

    def __init__(
        self,
        feature_columns: List[str],
        fixed_labels: List[str],
        variable_labels: List[str],
        perm_or_comb: TypePairing = "perm",
        combo_count: int = 2,
    ) -> None:
        """
        Args:
            feature_columns (List[str]): List of column names containing the features. These columns will be 
            concatenated horizontally in the new row. Should have different values for each row
            fixed_labels (List[str]): List of column names containing the labels that will be kept constant. These
            columns contain the save value for each row. In other words, the rows are grouped based on this column
            variable_labels (List[str]): List of column names containing the labels that will be permuted/combined
            perm_or_comb (Literal['perm', 'comb', 'perm_r', 'comb_r']): Whether to use Permutation or Combination when
            generating point pairs from the data points in the same group
            combo_count (int): How many rows to pick from the same group to create the new row
        """
        super().__init__()
        self.feature_columns = feature_columns
        self.fixed_labels = fixed_labels
        self.variable_labels = variable_labels
        self.perm_or_comb: TypePairing = perm_or_comb
        self.combo_count = combo_count
        self._label = []

    def build_feature(self, data: DataFrame) -> DataFrame:
        data = self._apply_chain(data)
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
        column_ids = product([1, 2], self.feature_columns)
        return [f"{feature_name}_{n}" for n, feature_name in column_ids]

    def get_label_names(self) -> List[str]:
        new_variable_columns = []
        for i in range(self.combo_count):
            new_variable_columns.append(*[f"{var} {i}" for var in self.variable_labels])

        # DO NOT change this ordering
        return self.fixed_labels + new_variable_columns

    @classmethod
    def from_chain(
        cls,
        chain: DataTransformation,
        variable_labels: List[str],
        perm_or_comb: TypePairing = "perm",
        combo_count: int = 2,
    ) -> "RowCombinationFeatureBuilder":
        fixed_labels = [label for label in chain.get_label_names() if label not in variable_labels]
        return_obj = cls(chain.get_feature_names(), fixed_labels, variable_labels, perm_or_comb, combo_count)
        return_obj.chain = chain
        return return_obj


class FetalACFeatureBuilder(FeatureBuilder):
    """
    Creates the AC component of the intensity at each SDD for the same simulation parameter combinations by taking the
    difference between intensity.
    """

    def __init__(
        self,
        conc_group_column: str,
        perm_or_comb: TypePairing,
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
            perm_or_comb (Literal['perm', 'comb', 'perm_r', 'comb_r']): Whether to use Permutation or Combination when generating AC points
            from the data points in the same group
            spatial_intensity_columns (List[str]): List of column names containing the spatial intensity values
            labels (List[str]): List of column names containing the simulation parameters (Stays untouched)
        """
        super().__init__()
        self.conc_group_column = conc_group_column
        self.mode = mode
        self.perm_or_comb: TypePairing = perm_or_comb
        self._label_names = labels
        self.intensity_columns = spatial_intensity_columns  # Sorted Wv1 first then Wv2
        fixed_labels = self._create_combination_index_columns()
        self.combination_feature_builder = RowCombinationFeatureBuilder(
            self.intensity_columns, fixed_labels, ["Fetal Hb Concentration"], self.perm_or_comb, 2
        )

    def build_feature(self, data: DataFrame) -> DataFrame:
        data = self._apply_chain(data)
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

    @classmethod
    def from_chain(
        cls,
        chain: DataTransformation,
        conc_group_column: str,
        perm_or_comb: TypePairing,
        mode: Literal["-", "/"],
    ) -> "FetalACFeatureBuilder":
        return_obj = cls(conc_group_column, perm_or_comb, mode, chain.get_feature_names(), chain.get_label_names())
        return_obj.chain = chain
        return return_obj

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
        perm_or_comb: TypePairing,
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
        self.perm_or_comb: TypePairing = perm_or_comb
        self.dc_mode = dc_mode
        self._feature_names = []
        self._label_names = labels
        self.intensity_columns = spatial_intensity_columns  # Sorted Wv1 first then Wv2
        fixed_labels = self._create_combination_index_columns()
        self.combination_feature_builder = RowCombinationFeatureBuilder(
            self.intensity_columns, fixed_labels, ["Fetal Hb Concentration"], self.perm_or_comb, 2
        )

    def build_feature(self, data: DataFrame) -> DataFrame:
        data = self._apply_chain(data)
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
            combination_data[_feature_names[i]] = np.divide(numerator_wv1, denominator_wv1)
            combination_data[_feature_names[i + detector_count]] = np.divide(numerator_wv2, denominator_wv2)

        # Clean-up intermediate columns
        combination_data.drop(columns=combination_feature_names, inplace=True)
        return combination_data

    def get_one_word_description(self) -> str:
        return "Fetal\nACbyDC"

    def get_feature_names(self) -> List[str]:
        detector_sdd = self._get_detector_sdd()
        feature_name_appender = "MIN" if self.dc_mode == "min" else "MAX"
        # DO NOT Change this ordering
        id_pairs = product(range(1, WAVE_INT_COUNT + 1), detector_sdd)
        return [f"{feature_name_appender}_ACbyDC_WV{wave_int}_{sdd}" for wave_int, sdd in id_pairs]

    def get_label_names(self) -> List[str]:
        return self.combination_feature_builder.get_label_names()

    def _get_detector_sdd(self) -> List[str]:
        """
        Get the SDD of the detectors using the intensity columns. Assumes the intensity columns
        are formatted as "SDD_WaveInt" and all the SDDs are the same for each wave int and are
        place consecutively in the list
        """
        all_sdd = []
        for column in self.intensity_columns:
            sdd = column.split("_")[0]
            if sdd not in all_sdd:
                all_sdd.append(sdd)
        return all_sdd

    def _create_combination_index_columns(self) -> List[str]:
        index_columns = self._label_names.copy()
        index_columns.remove("Fetal Hb Concentration")
        if self.conc_group_column not in index_columns:
            index_columns.append(self.conc_group_column)
        return index_columns

    @classmethod
    def from_chain(
        cls,
        chain: DataTransformation,
        conc_group_column: str,
        perm_or_comb: TypePairing,
        dc_mode: Literal["max", "min"] = "min",
    ) -> FeatureBuilder:
        return_obj = cls(conc_group_column, perm_or_comb, chain.get_feature_names(), chain.get_label_names(), dc_mode)
        return_obj.chain = chain
        return return_obj

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
        super().__init__()
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
        data = self._apply_chain(data)
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
        old_features_remaining = [
            feature
            for feature in self.old_features
            if (feature not in self.term1_cols) and (feature not in self.term2_cols)
        ]
        return old_features_remaining + self.new_features

    def get_label_names(self) -> List[str]:
        return self._labels

    def get_one_word_description(self) -> str:
        return "Division\nFeature"

    @classmethod
    def from_chain(
        cls,
        chain: DataTransformation,
        term1_cols: List[str],
        term2_cols: List[str],
        operator: Literal["+", "-", "*", "/"],
        keep_original: bool,
    ) -> "TwoColumnOperationFeatureBuilder":
        return_obj = cls(
            term1_cols, term2_cols, operator, keep_original, chain.get_feature_names(), chain.get_label_names()
        )
        return_obj.chain = chain
        return return_obj

    def _generate_new_feature_names(self) -> List[str]:
        feature_names = [f"{term1}_{self.operator}_{term2}" for term1, term2 in zip(self.term1_cols, self.term2_cols)]
        return feature_names


class LogTransformFeatureBuilder(FeatureBuilder):
    """Apply log transformation to the columns_to_log"""

    def __init__(self, columns_to_log: List[str], feature_columns: List[str], labels: List[str]) -> None:
        super().__init__()
        self.columns_to_log = columns_to_log
        self.feature_columns = feature_columns
        self._labels = labels

    def build_feature(self, data: DataFrame) -> DataFrame:
        data = self._apply_chain(data)
        # Check if the columns to log exist in the data
        if not all([col in data.columns for col in self.columns_to_log]):
            raise ValueError("Columns to log do not exist in the data")
        new_data = data.copy()
        new_data[self.columns_to_log] = np.log10(new_data[self.columns_to_log])
        return new_data

    def get_one_word_description(self) -> str:
        return "Log\nTransform"

    def get_feature_names(self) -> List[str]:
        return self.feature_columns.copy()

    def get_label_names(self) -> List[str]:
        return self._labels.copy()

    @classmethod
    def from_chain(cls, chain: DataTransformation, columns_to_log: List[str]) -> "LogTransformFeatureBuilder":
        return_obj = cls(columns_to_log, chain.get_feature_names(), chain.get_label_names())
        return_obj.chain = chain
        return return_obj


class ConcatenateFeatureBuilder(FeatureBuilder):
    """
    Concatenate the features from multiple FeatureBuilders. If there are any duplicate feature names, all the feature
    names are replaced by "c_n" where n is the some integer
    """

    def __init__(self, feature_builders: List[FeatureBuilder]):
        super().__init__()
        self.feature_builders = feature_builders
        self.rename_features = False
        if not self._check_labels_are_same():
            raise ValueError("Feature builders do not have the same labels")
        if not self._check_features_are_unique():
            self.rename_features = True

    def build_feature(self, data: DataFrame) -> DataFrame:
        data = self._apply_chain(data)
        transformed_data = [feature_builder(data) for feature_builder in self.feature_builders]
        data_labels_np = transformed_data[0][self.feature_builders[0].get_label_names()].to_numpy()
        data_features = [
            feature_builder_data[feature_builder.get_feature_names()].to_numpy()
            for feature_builder, feature_builder_data in zip(self.feature_builders, transformed_data)
        ]
        data_np = np.hstack([data_labels_np, *data_features])

        return DataFrame(data=data_np, columns=self.get_label_names() + self.get_feature_names())

    def get_one_word_description(self) -> str:
        return "Concatenate"

    def get_label_names(self) -> List[str]:
        return self.feature_builders[0].get_label_names()

    def get_feature_names(self) -> List[str]:
        if self.rename_features:
            total_feature_count = sum(
                [len(feature_builder.get_feature_names()) for feature_builder in self.feature_builders]
            )
            feature_column_names = [f"c_{i}" for i in range(total_feature_count)]
        else:
            feature_column_names = [feature_builder.get_feature_names() for feature_builder in self.feature_builders]
            feature_column_names = [
                feature_name for feature_names in feature_column_names for feature_name in feature_names
            ]
        return feature_column_names

    @classmethod
    def from_chain(
        cls, chain: DataTransformation, feature_builders: List[FeatureBuilder]
    ) -> "ConcatenateFeatureBuilder":
        raise NotImplementedError("ConcatenateFeatureBuilder cannot be created from a chain yet!")

    def _check_labels_are_same(self):
        """
        Check if the feature builders have the same labels. Returns True if all the labels are the same
        """
        # Make sure all the feautre builders have the same labels
        labels = self.feature_builders[0].get_label_names()
        for feature_builder in self.feature_builders[1:]:
            if any([label not in feature_builder.get_label_names() for label in labels]):
                return False
        return True

    def _check_features_are_unique(self):
        """
        Check if the feature builders have unique feature names
        """
        # Make sure all the feautre builders have the same labels
        feature_names = self.feature_builders[0].get_feature_names()
        for feature_builder in self.feature_builders[1:]:
            if any([feature_name in feature_builder.get_feature_names() for feature_name in feature_names]):
                return False
        return True
