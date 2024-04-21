import unittest
import pandas as pd
from pathlib import Path
from inverse_modelling_tfo.features.data_transformations import (
    LongToWideIntensityTransformation,
    ToFittingParameterTransformation,
)


TEST_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "intensity" / "s_based_intensity_low_conc.pkl"
DETECTOR_COUNT = 20
WAVE_INT_COUNT = 2


class TestLongToWideIntensityTransformation(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle(TEST_DATA_PATH)
        self.data_transformer = LongToWideIntensityTransformation()
        self.transformed_data = self.data_transformer.transform(self.data)

    def test_transform_length(self):
        self.assertEqual(
            len(self.transformed_data),
            len(self.data) // (DETECTOR_COUNT * WAVE_INT_COUNT),
            "20 Det x 2 wavelengths, transformed data should be 1/40th length",
        )

    def test_columns_names_are_strings(self):
        self.assertTrue(all([isinstance(x, str) for x in self.transformed_data.columns]))

    def test_all_columns_exist(self):
        self.assertTrue(all([x in self.transformed_data.columns for x in self.data_transformer.get_feature_names()]))
        self.assertTrue(all([x in self.transformed_data.columns for x in self.data_transformer.get_label_names()]))

    def test_correct_feature_numer(self):
        expected_feature_count = DETECTOR_COUNT * WAVE_INT_COUNT
        self.assertEqual(len(self.data_transformer.get_feature_names()), expected_feature_count)


class TestToFittingParameterTransformation(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.read_pickle(TEST_DATA_PATH)
        self.data_transformer = ToFittingParameterTransformation()
        self.transformed_data = self.data_transformer.transform(self.data)

    def test_transform_length(self):
        self.assertEqual(
            len(self.transformed_data),
            len(self.data) // (DETECTOR_COUNT * WAVE_INT_COUNT),
            "20 Det x 2 wavelengths, transformed data should be 1/40th length",
        )

    def test_column_names_are_strings(self):
        self.assertTrue(all([isinstance(x, str) for x in self.transformed_data.columns]))

    def test_all_columns_exist(self):
        self.assertTrue(all([x in self.transformed_data.columns for x in self.data_transformer.get_feature_names()]))
        self.assertTrue(all([x in self.transformed_data.columns for x in self.data_transformer.get_label_names()]))


if __name__ == "__main__":
    unittest.main()
