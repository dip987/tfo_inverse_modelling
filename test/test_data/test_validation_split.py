import unittest

import pandas as pd
from inverse_modelling_tfo.model_training.validation_methods import CombineMethods, RandomSplit, HoldOneOut
from inverse_modelling_tfo.features.build_features_helpers import create_curve_fitting_param

data = pd.read_pickle(
    r"/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/s_based_intensity_low_conc2.pkl"
)

# Drop Uterus Thickness for now
data = data.drop(columns="Uterus Thickness")

# Get Interpolation parameters
data, _, __ = create_curve_fitting_param(data, (1.0, 0.8))

common_columns = [
    "Maternal Wall Thickness",
    "Fetal Saturation",
    "Maternal Saturation",
    "Maternal Hb Concentration",
    "Fetal Hb Concentration",
]
VAL_COLUMN = "Maternal Wall Thickness"
VAL_VALUE = 6.0
CV_SPLIT_COUNT = 5


class TestSplitOverlap(unittest.TestCase):
    def test_hold_one_out(self):
        val_method = HoldOneOut(VAL_COLUMN, VAL_VALUE)
        train, val = val_method.split(data)
        self.assertEqual(all(train[VAL_COLUMN] != VAL_VALUE), True, "Train split contains held out rows")
        self.assertEqual(all(val[VAL_COLUMN] == VAL_VALUE), True, "Validation split contains non-held rows")

    def test_random_split(self):
        val_method = RandomSplit()
        train, val = val_method.split(data)
        joined1 = train.merge(val, on=common_columns, how="outer")
        joined2 = train.merge(val, on=common_columns, how="inner")

        self.assertEqual(len(joined1), len(train) + len(val), "Train and Val split do not cover all the data points")
        self.assertEqual(len(joined2), 0, "Overlapping Rows between train and val split")

    # TODO: Change this testcase
    # def test_cv_split(self):
    #     splits = []
    #     for i in range(CV_SPLIT_COUNT):
    #         val_method = CVSplit(CV_SPLIT_COUNT, i)
    #         _, val = val_method.split(data)
    #         splits.append(val)

    #     joined1 = splits[0].merge(splits[1], on=common_columns, how="outer")
    #     joined2 = splits[0].merge(splits[1], on=common_columns, how="inner")
    #     for i in range(2, CV_SPLIT_COUNT):
    #         joined1 = joined1.merge(splits[i], on=common_columns, how="outer")
    #         joined2 = joined2.merge(splits[i], on=common_columns, how="inner")

    #     self.assertEqual(len(joined1), len(data), "All CV splits combined do not cover the orignal data points")
    #     self.assertEqual(len(joined2), 0, "There exists common rows between the CV splits")

    def test_combine_splits(self):
        val_method = CombineMethods([HoldOneOut(VAL_COLUMN, VAL_VALUE), RandomSplit()])
        train, val = val_method.split(data)
        joined1 = train.merge(val, on=common_columns, how="outer")
        joined2 = train.merge(val, on=common_columns, how="inner")

        self.assertEqual(len(joined1), len(train) + len(val), "Train and Val split do not cover all the data points")
        self.assertEqual(len(joined2), 0, "Overlapping Rows between train and val split")


if __name__ == "__main__":
    unittest.main()
