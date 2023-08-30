from torch.optim import Adam, SGD
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from inverse_modelling_tfo.models import train_model, train_model_wtih_reporting
from inverse_modelling_tfo.data import generate_data_loaders, equidistance_detector_normalization, constant_detector_count_normalization, generate_differential_data_loaders, DifferentialCombinationDataset
from inverse_modelling_tfo.data.intensity_interpolation import get_interpolate_fit_params_custom, interpolate_exp, interpolate_exp_transform
from inverse_modelling_tfo.data.interpolation_function_zoo import *
from inverse_modelling_tfo.models.custom_models import SplitChannelCNN, PerceptronReLU
from inverse_modelling_tfo.features.build_features import create_ratio, create_spatial_intensity
from inverse_modelling_tfo.misc.misc_training import set_seed
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torchinfo
# Set my GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, input, target):
        # Custom loss calculation
        loss = torch.mean(torch.abs(input - target)/target)  # Example: mean absolute error
        return loss
        



data = pd.read_pickle(r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/s_based_intensity_low_conc2.pkl')
equidistance_detector_normalization(data)

# Drop Uterus Thickness for now
data = data.drop(columns='Uterus Thickness')
experiment_sdd = [1.5, 3.0, 4.5, 7.0, 10.0]

# Interpolate simulation data to have same SDD as experiments
data = interpolate_exp_transform(data, experiment_sdd, [1.0, 0.8])

# Interpolate simulation data to have same SDD as experiments
# data = interpolate_exp_transform(data, experiment_sdd, [1.0, 0.8])

# Manual log(intensity) normalization
data['Intensity'] = np.log10(data['Intensity'])        # Far values wayy to small to affect anything. Take log
# data.head()
data = create_spatial_intensity(data)
## Y -> Target
# y_columns = ['Maternal Wall Thickness', "Maternal Hb Concentration", "Maternal Saturation", "Fetal Hb Concentration", "Fetal Saturation"]
# y_columns = ['Maternal Saturation']
# y_columns = ['Maternal Hb Concentration']
# y_column = 'Fetal Hb Concentration'
y_column = 'Fetal Saturation'
fixed_columns = ['Maternal Wall Thickness', "Maternal Hb Concentration", "Maternal Saturation", "Fetal Hb Concentration"]
# Just to make sure I don't write stupid code
assert(len(fixed_columns) == 4), "4/5 TMPs should remain fixed. fixed_column length should be 4"
assert(y_column not in fixed_columns), "y_column is the TMP that changes. It cannot be inside fixed_columns"

# y_columns = ['Fetal Hb Concentration']

## X -> Predictors
# x_columns = list(filter(lambda X: '_' in X, data.columns))
# x_columns = list(filter(lambda X: X.isdigit(), data.columns))
x_columns = list(filter(lambda X: X.isdigit(), data.columns)) + list(filter(lambda X: '_' in X, data.columns))



## Pass in maternal info
# x_columns += ["Maternal Hb Concentration", "Maternal Saturation"]

## Scale y
y_scaler = preprocessing.StandardScaler()
data[y_column] = y_scaler.fit_transform(data[y_column].to_numpy().reshape(-1, 1))

## Scale x
# I tried using the same scaling for all to preserve spatial information. Training does not work
# With variable scale the network learns how much weight to give each 
x_scaler = preprocessing.StandardScaler()
data[x_columns] = x_scaler.fit_transform(data[x_columns])
## Manual scale - if needed (With maternal info.)
# data[x_columns[:-2]] /= 100.0    # stddev.   (Actual value is higher but let's keep it here for now)
# data[x_columns[:-2]] += 0.5  # unit var, 0 mean

## Scale non-intensity x columns (Maternal Hb Conc. , Maternal Saturation)
# data["Maternal Saturation"] -= 0.5 
# data["Maternal Hb Concentration"] /= 20
# data["Maternal Hb Concentration"] -= 0.5 


IN_FEATURES = len(x_columns) * 2
OUT_FEATURES = 1
model_config = {
    # 'model_class' : SplitChannelCNN,  # Class name
    'model_class' : PerceptronReLU,  # Class name
    # 'model_params' :  [2, IN_FEATURES, 4, 5, [2, OUT_FEATURES]],    # Input params as an array
    # 'model_params' :  [3, IN_FEATURES, 6, 5, [6, 3, OUT_FEATURES]],    # Input params as an array
    # 'model_params' :  [3, IN_FEATURES, 6, 7, [3, OUT_FEATURES]],    # Input params as an array
    # 'model_params' :  [4, IN_FEATURES, 8, 7, [4, 2, OUT_FEATURES]],    # Input params as an array
    # 'model_params' :  [[IN_FEATURES, 20, 8, OUT_FEATURES]],    # Input params as an array
    'model_params' :  [[IN_FEATURES, 10, 4, OUT_FEATURES]],    # Input params as an array
    'train_split' : 0.8,
    'epochs' : 25,
    # 'total_data_len': 120000,
    'total_data_len': 500,
    'allow_zero_diff': False,
    'hyperparam_search_count': 20,
    'hyperparam_max_epoch': 10,
    'seed': 42
}

params = {
    'batch_size': 32, 'shuffle': True, 'num_workers': 2
}

set_seed(model_config['seed'])
params = {
    'batch_size': 32, 'shuffle': True, 'num_workers': 2
}
# train, val = generate_data_loaders(data, params, x_columns, y_columns, model_config['train_split'])
train, val = generate_differential_data_loaders(data, params, fixed_columns, x_columns, y_column, model_config['total_data_len'], model_config["allow_zero_diff"], model_config['train_split'])
# model = create_perceptron_model(config['model'])
# model = create_perceptron_model([42, 8, 1])
# model = TwoChannelCNN(40, 4, 5, [4, 1])
model = model_config['model_class'](*model_config['model_params'])
# criterion = nn.MSELoss()
criterion = CustomLoss()
optimizer = SGD(model.parameters(), lr=3e-4, momentum=0.95)
# optimizer = Adam(model.parameters(), lr=config["lr"], betas=[config["b1"], config["b2"]])
train_loss, val_loss = train_model(model, optimizer=optimizer, criterion=criterion, train_loader=train, validation_loader=val, epochs=4)
