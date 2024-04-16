import sys

# Set path to root directory
sys.path.append(r'/home/rlfowler/Documents/research/tfo_inverse_modelling')

import os
from pathlib import Path
import numpy as np
import pandas as pd
from inverse_modelling_tfo.data import config_based_normalization
from inverse_modelling_tfo.features.build_features import FetalACbyDCFeatureBuilder
from inverse_modelling_tfo.features.data_transformations import LongToWideIntensityTransformation


# Set my GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

DATA_PATH = r'/home/rlfowler/Documents/research/tfo_inverse_modelling/Randalls Folder/data/randall_data_intensities.pkl'
# CONFIG_PATH = Path(r'/home/rlfowler/Documents/research/tfo_sim/data/compiled_intensity/randall_data.json')

# Load data
data = pd.read_pickle(DATA_PATH)

# # Normalize data using the json file
# config_based_normalization(data, CONFIG_PATH) # May need to change this for my own code

# # # Drop Uterus Thickness for now
# data = data.drop(columns="Uterus Thickness")

# data_transformer = LongToWideIntensityTransformation()
# data.dropna(inplace=True)

def get_FconcCenters(fetal_conc):
    f_c_range = (11., 16.)
    count = 11
    perc = 0.026
    fetal_concentrations = np.linspace(f_c_range[0], f_c_range[1], count, endpoint=True)
    FconcCenters = np.ones(fetal_conc.shape, dtype=np.float32)*np.Inf

    for i, f_c in enumerate(fetal_concentrations): 
        FconcCenters[np.bitwise_and(fetal_conc >= np.round(f_c*(1-perc),4), fetal_conc <= np.round(f_c*(1+perc),4))] = i
    return FconcCenters

# fetal_conc_group_mapping = get_fetal_concetration_group()
# print(fetal_conc_group_mapping)

data['FconcCenters'] = get_FconcCenters(data['Fetal Hb Concentration'])

# data = data_transformer.transform(data)
# labels = data_transformer.get_label_names()
# intensity_columns = data_transformer.get_feature_names()

intensity_columns = data.columns[7:].tolist()
labels = data.columns[0:7].tolist()

fb1 = FetalACbyDCFeatureBuilder('FconcCenters', 'perm', intensity_columns, labels, "max")
data = fb1(data)

data.to_pickle('randall_data_ACDC_max.pkl')
