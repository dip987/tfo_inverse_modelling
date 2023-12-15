import unittest
from inverse_modelling_tfo.features.build_features import *


class TestConcateFeatureBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.fb1 = TwoColumnOperationFeatureBuilder(["a", "b"], ["c", "d"], "+", True, ["a", "b", "c", "d"], ["Label"])
        self.fb2 = TwoColumnOperationFeatureBuilder(["e", "f"], ["g", "h"], "+", True, ["e", "f", "g", "h"], ["Label"])
        self.fb_bad = TwoColumnOperationFeatureBuilder(["e", "f"], ["g", "h"], "+", True, ["e", "f", "g", "h"], ["Yeet"])
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


if __name__ == "__main__":
    unittest.main()
