"""
Produces Interpolated Pulsation Ratio(I2/I1) using stable derivatives
"""

"""
Pipe line to process simulation data and extract features for the inverse modelling task, and save them for future use.
The purpose is to cut down on the time it takes to process the data and extract features.
"""

from pathlib import Path
import json
import time
import pandas as pd
from inverse_modelling_tfo.data_pipelines.fetal_conc_groups import dan_iccps_pencil1, generate_grouping_from_config
from inverse_modelling_tfo.data import config_based_normalization
from inverse_modelling_tfo.data.interpolation_function_zoo import *
from inverse_modelling_tfo.features.build_features import (
    FetalACFeatureBuilder,
    RowCombinationFeatureBuilder,
    TwoColumnOperationFeatureBuilder,
    FetalACbyDCFeatureBuilder,
    LogTransformFeatureBuilder,
    ConcatenateFeatureBuilder,
)
from inverse_modelling_tfo.features.data_transformations import LongToWideIntensityTransformation
from inverse_modelling_tfo.data_pipelines.stable_derivative import interpolate_pr

# Data Setup
# ==========================================================================================
out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "pulsation_ratio_interp_sd2.pkl"

config_dest = out_dest.with_suffix(".json")

# in_src = Path(r"/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/pencil2.pkl")
in_src = Path(r"/home/rraiyan/personal_projects/tfo_inverse_modelling/data/processed_data/I1_and_I2.pkl")
config_src = in_src.with_suffix(".json")

data = pd.read_pickle(in_src)

# Cleanup
data.dropna(inplace=True)

# Create AC/DC as log(I1)/log(I2)
# Columns that stay the same between two combinations
with open(config_src, "r", encoding="utf-8") as infile:
    config = json.load(infile)
combinations_features = config["features"]
labels = config["labels"]


fb1 = TwoColumnOperationFeatureBuilder(
    combinations_features[: len(combinations_features) // 2],  # I1
    combinations_features[len(combinations_features) // 2 :],  # I2
    "/",
    False,
    combinations_features,
    labels,
)


# Build features
data = fb1(data)

# Interpolate PR
ma_filter_len = 2
derivative_threshold = 1e-4

wv1_columns = [col for col in fb1.get_feature_names() if "_1.0_" in col]
wv2_columns = [col for col in fb1.get_feature_names() if "_2.0_" in col]


def apply_interpolation(x) -> np.ndarray:
    return interpolate_pr(
        x,
        derivative_threshold,
        ma_filter_len,
    )


t1 = np.apply_along_axis(apply_interpolation, 1, data[wv1_columns].values)
t2 = np.apply_along_axis(apply_interpolation, 1, data[wv2_columns].values)
data[wv1_columns] = t1
data[wv2_columns] = t2

# Create Config file
# ==========================================================================================
# NOT AUTOGENRATED! MUST BE DONE MANUALLY FOR EACH PIPELINE
config = {
    "labels": fb1.get_label_names(),
    "features": fb1.get_feature_names(),
    "feature_builder_txt": str(fb1),
    "preprocessing_description": "Detector Normalization -> Long to Wide -> Row Combination -> Calculate I2/I1 -> Interpolate PR",
    "comments": "Stable derviative method used to interpolate PR (Using a MA filter of length 2)",
    "data used": "/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/pencil2.pkl",
}

# Save data and config
# ==========================================================================================
data.to_pickle(out_dest)

with open(config_dest, "w+", encoding="utf-8") as outfile:
    json.dump(config, outfile)
