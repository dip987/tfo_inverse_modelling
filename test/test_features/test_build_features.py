import unittest
import pandas as pd
from inverse_modelling_tfo.features.build_features import *


class TestTwoColumnOperationFeatureBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 4, 5], "Label": [0.8, 0.9, 1.0]})

    def test_addition(self):
        fb = TwoColumnOperationFeatureBuilder(["A"], ["B"], "+", True, ["A", "B"], ["Label"])
        df = fb(self.df)
        self.assertTrue(all(df["A_+_B"] == df["A"] + df["B"]))
        self.assertTrue(all([expect_column in df.columns for expect_column in ["A", "B", "A_+_B", "Label"]]))

    def test_subtraction(self):
        fb = TwoColumnOperationFeatureBuilder(["A"], ["B"], "-", True, ["A", "B"], ["Label"])
        df = fb(self.df)
        self.assertTrue(all(df["A_-_B"] == df["A"] - df["B"]))
        self.assertTrue(all([expect_column in df.columns for expect_column in ["A", "B", "A_-_B", "Label"]]))

    def test_multiplication(self):
        fb = TwoColumnOperationFeatureBuilder(["A"], ["B"], "*", True, ["A", "B"], ["Label"])
        df = fb(self.df)
        self.assertTrue(all(df["A_*_B"] == df["A"] * df["B"]))
        self.assertTrue(all([expect_column in df.columns for expect_column in ["A", "B", "A_*_B", "Label"]]))

    def test_division(self):
        fb = TwoColumnOperationFeatureBuilder(["A"], ["B"], "/", True, ["A", "B"], ["Label"])
        df = fb(self.df)
        self.assertTrue(all(df["A_/_B"] == df["A"] / df["B"]))
        self.assertTrue(all([expect_column in df.columns for expect_column in ["A", "B", "A_/_B", "Label"]]))


class TestLogTransformFeatureBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({"A": [1, 1, 2], "Label": [0.8, 0.9, 1.0]})

    def test_log_transform(self):
        fb = LogTransformFeatureBuilder(["A"], ["A"], ["Label"])
        transformed_df = fb(self.df)
        self.assertTrue(all(transformed_df["A"] == np.log10(self.df["A"])))


class TestConcateFeatureBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.fb1 = TwoColumnOperationFeatureBuilder(["a", "b"], ["c", "d"], "+", True, ["a", "b", "c", "d"], ["Label"])
        self.fb2 = TwoColumnOperationFeatureBuilder(["e", "f"], ["g", "h"], "+", True, ["e", "f", "g", "h"], ["Label"])
        self.fb_bad = TwoColumnOperationFeatureBuilder(
            ["e", "f"], ["g", "h"], "+", True, ["e", "f", "g", "h"], ["Yeet"]
        )
        self.fb1_alt = TwoColumnOperationFeatureBuilder(
            ["a", "b"], ["c", "d"], "+", True, ["a", "b", "c", "d"], ["Label"]
        )

    def test_concat_feature_names(self):
        fb_concat = ConcatenateFeatureBuilder([self.fb1, self.fb2])
        self.assertCountEqual(
            self.fb1.get_feature_names() + self.fb2.get_feature_names(), fb_concat.get_feature_names()
        )

    def test_concat_label_names(self):
        fb_concat = ConcatenateFeatureBuilder([self.fb1, self.fb2])
        self.assertCountEqual(self.fb1.get_label_names(), fb_concat.get_label_names())
        self.assertCountEqual(self.fb2.get_label_names(), fb_concat.get_label_names())

    def test_feature_rename_for_name_collision(self):
        fb_concat = ConcatenateFeatureBuilder([self.fb1, self.fb1_alt])
        name_similarity = any([f_name in self.fb1.get_feature_names() for f_name in fb_concat.get_feature_names()])
        self.assertFalse(name_similarity)

    def test_non_similar_label_raises_error(self):
        with self.assertRaises(ValueError):
            _ = ConcatenateFeatureBuilder([self.fb1, self.fb_bad])


class TestRowCombinationFeatureBuilder(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
