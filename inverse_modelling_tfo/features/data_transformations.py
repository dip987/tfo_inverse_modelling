from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from pandas.core.indexes.base import Index
from pandas import pivot, DataFrame
from inverse_modelling_tfo.data.intensity_interpolation import (
    InterpolatorFuncType,
    default_interpolator,
    get_interpolate_fit_params,
)


class DataTransformation(ABC):
    """Abstract class for transforming data. The data should have the following columns:
    - SDD: Source to Detector Distance
    - Wave Int: Wavelength Intensity
    - Intensity: Intensity value
    - Tissue Model parameters

    The feature and label names will only be available after transform has been called!
    Note: No transformation should mutate the original data.
    """

    def __init__(self):
        pass

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

    @abstractmethod
    def transform(self, data: DataFrame) -> DataFrame:
        """
        Transforms the data from one format to another. The original data should not be mutated.
        """

    def __call__(self, *args: Any, **kwds: Any) -> DataFrame:
        return self.transform(*args, **kwds)


class SimDataTransformation(DataTransformation):
    """Abstract class for transforming Simulation Data loaded directly from pickle files. The data should have the
    following columns:
    - SDD: Source to Detector Distance
    - Wave Int: Wavelength Intensity
    - Intensity: Intensity value
    - Tissue Model parameters

    The feature and label names will only be available after transform has been called!
    Note: No transformation should mutate the original data.
    """

    def verfiy_data_columns(self, data: DataFrame) -> bool:
        """
        Verifies that the data has the correct columns. Returns True if the data is valid, otherwise False.
        """
        required_columns = ["SDD", "Wave Int", "Intensity"]
        return all([col in data.columns for col in required_columns])


class LongToWideIntensityTransformation(SimDataTransformation):
    """Transforms long format Intensity data to wide format by pivoting on the Wave Int and SDD columns
    
    After transformation, the TMP columns will be the labels and the new Intensity columns will be the features
    """

    def __init__(self):
        self.sim_param_columns = []
        self.feature_names = []

    def transform(self, data: DataFrame) -> DataFrame:
        # Check that the data has the correct columns
        if not self.verfiy_data_columns(data):
            raise ValueError("Data does not have the correct columns")

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

    def get_one_word_description(self) -> str:
        return "Intensity Long\nto Wide"

    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()

    def get_label_names(self) -> List[str]:
        return self.sim_param_columns.copy()


class ToFittingParameterTransformation(SimDataTransformation):
    """Replace Intensity values with their correspodning fitting parameters and return a wide-format dataframe"""

    def __init__(
        self,
        weights: Tuple[float, float] = (1.0, 0.6),
        sdd_chunk_size: int = 20,
        custom_fit_func: InterpolatorFuncType = default_interpolator,
        **interpolation_func_kwargs
    ):
        self.weights = weights
        self.sdd_chunk_size = sdd_chunk_size
        self.custom_fit_func = custom_fit_func
        self.interpolation_func_kwargs = interpolation_func_kwargs
        self.sim_param_columns = []
        self.feature_names = []

    def transform(self, data: DataFrame) -> DataFrame:
        # Check that the data has the correct columns
        if not self.verfiy_data_columns(data):
            raise ValueError("Data does not have the correct columns")

        new_data = data.copy()
        self.sim_param_columns = _get_sim_param_columns(new_data.columns)
        # Fitting Parameters for each [wave int, TMP]
        new_data = get_interpolate_fit_params(
            new_data, self.weights, self.sdd_chunk_size, self.custom_fit_func, **self.interpolation_func_kwargs
        )
        # Pivot to wide format for wave int
        fitting_param_columns = list(filter(lambda x: "alpha" in x, new_data.columns))
        new_data = pivot(
            new_data, index=self.sim_param_columns, columns="Wave Int", values=fitting_param_columns
        ).reset_index()
        # Flatten Index
        new_data.columns = [
            "_".join([str(col[0]), str(col[1])]) if col[1] != "" else col[0] for col in new_data.columns
        ]
        self.feature_names = list(filter(lambda x: "alpha" in x, new_data.columns))
        return new_data

    def get_one_word_description(self) -> str:
        return "Replace With\nFitting Param"

    def get_feature_names(self) -> List[str]:
        return self.feature_names.copy()

    def get_label_names(self) -> List[str]:
        return self.sim_param_columns.copy()


def _get_sim_param_columns(column_names: Index) -> List:
    result = column_names.copy().drop(["SDD", "Intensity", "Wave Int"], errors="ignore")
    return result.to_list()
