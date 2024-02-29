"""
Pipe line to process simulation data and extract features for the inverse modelling task, and save them for future use.
The purpose is to cut down on the time it takes to process the data and extract features.
"""

from pathlib import Path
import json
from tokenize import group
from webbrowser import get
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
out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "processed1_min_long_range.pkl"
config_dest = out_dest.with_suffix(".json")

# in_src = r'/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/dan_iccps_pencil.pkl'
in_src = Path(r'/home/rraiyan/simulations/tfo_sim/data/compiled_intensity/weitai_data.pkl')
config_src = in_src.with_suffix(".json")

fconc_rounding = 2
grouping_map = generate_grouping_from_config(config_src, fconc_rounding)


data = pd.read_pickle(in_src)
config_based_normalization(data, config_src)


# Data Processing
# ==========================================================================================
data = data.drop(columns="Uterus Thickness")

# Interpolate intensity to remove noise
# data = interpolate_exp(data, weights=(1, 0.6), interpolation_function=exp_piecewise_affine, break_indices=[4, 12, 20])
# data['Intensity'] = data['Interpolated Intensity']
# data = data.drop(columns='Interpolated Intensity')

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
data['FconcCenters'] = data['Fetal Hb Concentration'].round(fconc_rounding).map(grouping_map)
# fitting_params['FconcCenters'] = data['FconcCenters']

# Define Feature builders
fb1 = FetalACbyDCFeatureBuilder('FconcCenters', 'comb', intensity_columns, labels, "min")

# Build features
data = fb1(data)


# Create Config file
# ==========================================================================================
# NOT AUTOGENRATED! MUST BE DONE MANUALLY FOR EACH PIPELINE
config = {
    'labels' : fb1.get_label_names(),
    'features' : fb1.get_feature_names(),
    'preprocessing_description' : "Detector Normalization -> Long to Wide -> AC by DC(comb, min)",
    "comments": "Large range of fetal depths, from 1.0 cm to 4.8cm. Will be used to test if the range of depth is a \
    bottleneck for the model."
}

# Save data and config
# ==========================================================================================
data.to_pickle(out_dest)
with open(config_dest, "w+") as outfile: 
    json.dump(config, outfile)
    
