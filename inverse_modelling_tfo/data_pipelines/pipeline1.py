"""
Pipe line to process simulation data and extract features for the inverse modelling task, and save them for future use.
The purpose is to cut down on the time it takes to process the data and extract features.
"""

from pathlib import Path
import json
import pandas as pd
from inverse_modelling_tfo.data_pipelines.fetal_conc_groups import dan_iccps_pencil1, generate_grouping_from_config
from inverse_modelling_tfo.data import config_based_normalization
from inverse_modelling_tfo.data.intensity_interpolation import (
    interpolate_exp,
    get_interpolate_fit_params,
    exp_piecewise_affine,
)
from inverse_modelling_tfo.data.interpolation_function_zoo import *
from inverse_modelling_tfo.features.build_features import (
    FetalACFeatureBuilder,
    RowCombinationFeatureBuilder,
    TwoColumnOperationFeatureBuilder,
    FetalACbyDCFeatureBuilder,
    LogTransformFeatureBuilder,
    ConcatenateFeatureBuilder,
)
from inverse_modelling_tfo.features.data_transformations import (
    LongToWideIntensityTransformation,
    ToFittingParameterTransformation,
)

# Data Setup
# ==========================================================================================
# out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "I1_and_I2.pkl"
out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "pulsation_ratio_interp.pkl"
# out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "logI2_by_I1.pkl"
# out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "processed1_max_long_range.pkl"
config_dest = out_dest.with_suffix(".json")

in_src = Path(r"/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/pencil2.pkl")
config_src = in_src.with_suffix(".json")

fconc_rounding = 2
grouping_map = generate_grouping_from_config(config_src, fconc_rounding)


data = pd.read_pickle(in_src)
config_based_normalization(data, config_src)


# Data Processing
# ==========================================================================================
data = data.drop(columns="Uterus Thickness")

# Interpolate intensity to remove noise
# IF YOU WANT TO INTERPOLATE LARGE DATASET DO NOT USE THIS - USE TEST_PIPLINE2.ipynb
# data = interpolate_exp(data, weights=(1, 0.6), interpolation_function=exp_piecewise_affine, break_indices=[4, 12, 20])
# data["Intensity"] = data["Interpolated Intensity"]  # Replace OG intensity with interpolated intensity
# data = data.drop(columns="Interpolated Intensity")  # Cleanup

# Define data transformers
data_transformer = LongToWideIntensityTransformation()
# fitting_param_transformer = ToFittingParameterTransformation()

# Transform data
# fitting_params = fitting_param_transformer.transform(data)
data = data_transformer.transform(data)
labels = data_transformer.get_label_names()
intensity_columns = data_transformer.get_feature_names()

# Cleanup
data.dropna(inplace=True)

# Create fetal conc. grouping column - used for generating the AC component/which rows to choose for pairing
data["FconcCenters"] = data["Fetal Hb Concentration"].round(fconc_rounding).map(grouping_map)
labels = labels + ["FconcCenters"]  # This new column grouping should also be treated as a label
# fitting_params['FconcCenters'] = data['FconcCenters']
fixed_columns = [
    "Maternal Wall Thickness",
    "Maternal Hb Concentration",
    "Maternal Saturation",
    "Fetal Saturation",
    "FconcCenters",
]

# Define Feature builders
# Path 1
# Create AC/DC using (I1 - I2)/max(I1, I2) or min(I1, I2)
# fb1 = FetalACbyDCFeatureBuilder("FconcCenters", "comb", intensity_columns, labels, "max")

# Path 2
# Create AC/DC as log(I1)/log(I2)
# Columns that stay the same between two combinations
# # Apply log
# fb_log = LogTransformFeatureBuilder(intensity_columns, intensity_columns, labels)
# data = fb_log(data)
# fb0 = RowCombinationFeatureBuilder(intensity_columns, fixed_columns, ["Fetal Hb Concentration"], "comb")
# # Apply log(I2) / log(I1)
# combinations_features = fb0.get_feature_names()
# fb1 = TwoColumnOperationFeatureBuilder.from_chain(
#     fb0,
#     combinations_features[: len(combinations_features) // 2],
#     combinations_features[len(combinations_features) // 2 :],
#     "/",
#     False,
# )

# Path 3
# Apply Row combinations - place the 2 sets of I's reponsbile for calculating AC along a single row
fb1 = RowCombinationFeatureBuilder(intensity_columns, fixed_columns, ["Fetal Hb Concentration"], "comb")


# Build features
data = fb1(data)


# Create Config file
# ==========================================================================================
# NOT AUTOGENRATED! MUST BE DONE MANUALLY FOR EACH PIPELINE
config = {
    "labels": fb1.get_label_names(),
    "features": fb1.get_feature_names(),
    "feature_builder_txt": str(fb1),
    "preprocessing_description": "Detector Normalization -> Long to Wide -> Row Combination -> Calculate log(I2)/log(I1)",
    "comments": "Most vanilla data pipeline/feature extraction. No normalization, no feature engineering. Just the raw data reshaped for the model.",
    "data used": "/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/pencil2.pkl",
}

# Save data and config
# ==========================================================================================
data.to_pickle(out_dest)

with open(config_dest, "w+", encoding="utf-8") as outfile:
    json.dump(config, outfile)
