"""
Produces Interpolated Pulsation Ratio(I2/I1) using stable derivatives
"""

"""
Pipe line to process simulation data and extract features for the inverse modelling task, and save them for future use.
The purpose is to cut down on the time it takes to process the data and extract features.
"""

from pathlib import Path
import json
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
from inverse_modelling_tfo.data_pipelines.stable_derivative import interpolate_pr, interpolate_pr2

# Data Setup
# ==========================================================================================
out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "pulsation_ratio_interp_sd3_5wv.pkl"

config_dest = out_dest.with_suffix(".json")

# The in_src should be the output of the previous pipeline (pipeline1.py)
in_src = Path(r"/home/rraiyan/personal_projects/tfo_inverse_modelling/data/processed_data/I1_and_I2_5wv.pkl")
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

all_wavelengths = [x.split("_")[1] for x in combinations_features]
all_wavelengths = list(set(all_wavelengths))
all_wavelengths.sort()

# The feature names from fb1 have the format SDD_WV_1_/_SDD_WV_2
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
ma_filter_len = 3
derivative_threshold = 1e-4


pr_cols_split_per_wavelength = []       # A 2D list to store PR column names split by each wavelength
for wavelength_str in all_wavelengths:
    new_wv_cols = [col for col in fb1.get_feature_names() if col.split("_")[1] == wavelength_str]
    if new_wv_cols:
        pr_cols_split_per_wavelength.append(new_wv_cols)


def apply_interpolation(x) -> np.ndarray:
    return interpolate_pr2(
        x,
        derivative_threshold,
        ma_filter_len,
    )

# This applies the PR filter for each wavelength separately
for wv_cols in pr_cols_split_per_wavelength:
    data[wv_cols] = np.apply_along_axis(apply_interpolation, 1, data[wv_cols].values)

# Create Config file
# ==========================================================================================
# NOT AUTOGENRATED! MUST BE DONE MANUALLY FOR EACH PIPELINE
config = {
    "labels": fb1.get_label_names(),
    "features": fb1.get_feature_names(),
    "feature_builder_txt": str(fb1),
    "preprocessing_description": "Detector Normalization -> Long to Wide -> Row Combination -> Calculate I2/I1 -> Interpolate PR",
    "comments": "Has 5 Wavelength data. Stable derviative method used to interpolate PR (Using a MA filter of length 3)",
    "data used": "/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/pencil2_ext.pkl",
}

# Save data and config
# ==========================================================================================
data.to_pickle(out_dest)

with open(config_dest, "w+", encoding="utf-8") as outfile:
    json.dump(config, outfile)
