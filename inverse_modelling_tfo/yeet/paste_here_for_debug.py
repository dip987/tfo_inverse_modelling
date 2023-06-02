from torch.optim import Adam, SGD
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from inverse_modelling_tfo.models import train_model, create_perceptron_model, train_model_wtih_reporting
from inverse_modelling_tfo.data import generate_data_loaders, equidistance_detector_normalization, constant_detector_count_normalization
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params_custom
from inverse_modelling_tfo.data.interpolation_function_zoo import *
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

data = pd.read_pickle(r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/s_based_intensity.pkl')
equidistance_detector_normalization(data)

data['Intensity'] = np.log10(data['Intensity'])        # Far values wayy to small to affect anything. Take log
data = pd.pivot(data, index=['Maternal Wall Thickness', "Maternal Hb Concentration", "Maternal Saturation", "Fetal Hb Concentration", "Fetal Saturation"], columns=["SDD", "Wave Int"], values="Intensity").reset_index()
data.dropna(inplace=True)   # Slight coding mistake, not all waveints have both wv1 and 2
# Rename multi-index columns
data.columns = ['_'.join([str(col[0]), str(col[1])]) if col[1] != '' else col[0] for col in data.columns]
# y_columns = ['Maternal Wall Thickness', "Maternal Hb Concentration", "Maternal Saturation", "Fetal Hb Concentration", "Fetal Saturation"]
y_columns = ['Fetal Hb Concentration']
x_columns = list(filter(lambda X: '_' in X, data.columns))
# filtered_fitting_param_table = fitting_param_table[fitting_param_table['Wave Int'] == 2.0]
# x_scaler = preprocessing.StandardScaler()
y_scaler = preprocessing.StandardScaler()
data[y_columns] = y_scaler.fit_transform(data[y_columns])

# Manual log(intensity) normalization
data[x_columns] /= 100.0    # stddev.   (Actual value is higher but let's keep it here for now)
data[x_columns] += 0.5  # unit var : mean


model = create_perceptron_model([40, 1])
criterion = nn.MSELoss()
params = {
    'batch_size': 16, 'shuffle': True, 'num_workers': 2
}
optimizer = SGD(model.parameters(), lr=0.0009, momentum=1.0)
train, val = generate_data_loaders(data, params, x_columns, y_columns, 0.8)
train_loss, validation_loss = train_model(model, optimizer, criterion, train, val, epochs=30, gpu_to_use=3)

with torch.no_grad():
    x_data = torch.tensor(data[x_columns].values, dtype=torch.float).cuda()
    predictions = model(x_data)
    predictions = predictions.cpu().numpy().flatten()
    y_data = data[y_columns].to_numpy().flatten()
    error_df = pd.DataFrame({'Truth': y_data, "Predicted": predictions, "Absolute Error": np.abs(y_data - predictions)})
    print("HALT")
    