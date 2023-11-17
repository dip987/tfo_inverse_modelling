
import unittest

import numpy as np
import pandas as pd

from inverse_modelling_tfo.data.data_loader import *
from inverse_modelling_tfo.features.build_features import create_spatial_intensity

data = pd.read_pickle(
    r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/s_based_intensity_low_conc2.pkl')

# Drop Uterus Thickness for now
data = data.drop(columns='Uterus Thickness')
common_columns = ['Maternal Wall Thickness', "Fetal Saturation", "Maternal Saturation",
                  "Maternal Hb Concentration", 'Fetal Saturation', 'Fetal Hb Concentration']

# Manual log(intensity) normalization
# Far values wayy to small to affect anything. Take log
data['Intensity'] = np.log10(data['Intensity'])
data, _, __ = create_spatial_intensity(data)

dl = CustomDataset(data, ['10_1.0', '14_1.0'], ['Fetal Hb Concentration'])
dl = DifferentialCombinationDataset(data, ['Maternal Wall Thickness',
                                           "Fetal Saturation",
                                           "Maternal Saturation",
                                           "Maternal Hb Concentration"],
                                    x_columns=['10_1.0', '14_1.0'],
                                    differential_column='Fetal Hb Concentration',
                                    total_length=2000)
for i in range(40):
    a, b = dl[i]
    print('A shape')
    print(a.shape)
    print('B shape')
    print(b.shape)
# params = {}
# y_column = 'Fetal Saturation'
# fixed_columns = ['Maternal Wall Thickness',
#                  "Fetal Saturation", "Maternal Saturation"]
# x_columns = list(filter(lambda X: X.isdigit(), data.columns)) + \
#     list(filter(lambda X: '_' in X, data.columns))

# x_data, _ = generate_differential_data_loaders(data, params, fixed_columns, x_columns,
#                                                y_column, 2000, True, 1.0)
