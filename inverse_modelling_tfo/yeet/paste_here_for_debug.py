from itertools import product
from typing import List, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from TFO_dataset import SheepData
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params, get_interpolate_fit_params_custom
from inverse_modelling_tfo.data.interpolation_function_zoo import unity_at_zero_interpolation
from inverse_modelling_tfo.data.tfo_data_helpers import transform_tfo_data

# Load Data
tag = {'experiment_number': 5, 'experiment_round': 2, 'experiment_year_prefix': 'sp2022',
       'additional_info': '', 'data_version': 'iq_demod_optical'}
START_TIME_SECONDS = 50
END_TIME_SECONDS = 70
WEIGHTS = [0, 1]
data = SheepData('iq_demod_optical').get_data_from_tag(tag)
data = data.iloc[START_TIME_SECONDS * 80: END_TIME_SECONDS * 80, :]
print(f'Sample Length : {len(data)}')

sdd = SheepData('iq_demod_optical').get_sdd_distance(tag)
data = transform_tfo_data(data, sdd)
data.head()

fit_params = get_interpolate_fit_params_custom(
    data, unity_at_zero_interpolation, sdd_chunk_size=len(sdd), weights=WEIGHTS, return_alpha=True)
fit_params.head()
