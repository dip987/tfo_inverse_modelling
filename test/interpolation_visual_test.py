"""
Visually test out what the interpolation results should look like
(This does not test for bugs in the code itself. Use [interpolation_test] for that)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inverse_modelling_tfo.data.intensity_interpolation import *
from inverse_modelling_tfo.data.interpolation_function_zoo import *
from inverse_modelling_tfo.data.intensity_normalization import equidistance_detector_normalization

PATH = Path(__file__).resolve().parent.parent / "inverse_modelling_tfo" / "s_based_intensity_low_conc3.pkl"
data = pd.read_pickle(PATH)
equidistance_detector_normalization(data)

# data = interpolate_exp(data, weights=(1.0, 0.6))
data = interpolate_exp(
    data, weights=(1.0, 0.8), interpolation_function=exp_piecewise_affine, break_indices=[0, 4, 12, 20]
)
# len(break_indices) == piece_count - 1

data["Interpolation Error"] = np.abs(data["Intensity"] - data["Interpolated Intensity"])
data["Error %"] = data["Interpolation Error"] / data["Intensity"] * 100


# Describe the error for each SDD
# print(data.groupby("SDD")["Interpolation Error"].describe())
print(data.groupby("SDD")["Error %"].describe())

# plt.figure()
# data["Interpolation Error"].hist(bins=30
# plt.show()
