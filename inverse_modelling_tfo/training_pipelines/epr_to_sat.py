"""
Convert EPR to Fetal Saturation
"""

import os
from pathlib import Path
import json
import torch
from torch.optim.adam import Adam
import pandas as pd
from sklearn import preprocessing
from model_trainer import RandomSplit, HoldOneOut, ColumnBasedRandomSplit
from model_trainer import ModelTrainer, TorchLossWrapper, DataLoaderGenerator
import joblib
import torchinfo
from inverse_modelling_tfo.model_training.custom_models import PerceptronBD, SplitChannelCNN
from inverse_modelling_tfo.misc.misc_training import set_seed

# Set my GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# FILE_NAME = "pulsation_ratio_interp_sd3_6wv"  # EPR input - too lazy to change the name
FILE_NAME = "pulsation_ratio_interp_sd2_3wv(alt)" 
# FILE_NAME = "pulsation_ratio_interp_sd2_3wv(alt)"  # EPR input - too lazy to change the name
# Load data
PROJECT_BASE_PATH = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_BASE_PATH / "data" / "processed_data" / f"{FILE_NAME}.pkl"
data = pd.read_pickle(DATA_PATH)

# Load Configs
CONFIG_PATH = DATA_PATH.with_suffix(".json")
with open(CONFIG_PATH, "r", encoding="utf8") as f:
    config = json.load(f)
labels = config["labels"]
features = config["features"]

# wavelength_to_feature_map = {  # Which features correspond to which wavelengths - (forgot to sort before :(  )
#     910.0: features[:20],
#     690.0: features[20:40],
#     850.0: features[40:60],
#     735.0: features[60:80],
#     810.0: features[80:100],
#     780.0: features[100:120],
# }
wavelength_to_feature_map = {   # Settins for the older file
    735.0: features[:20],
    850.0: features[20:40],
    810.0: features[40:60],
}

wavelengths_to_use = [850.0, 735.0, 810.0]
features = [feature for wavelength in wavelengths_to_use for feature in wavelength_to_feature_map[wavelength]]

y_columns = ["Fetal Saturation"]
x_columns = features  # What to use as input
# x_columns = [features[i] for i in [3, 8, 12, 15, 19, 23, 28, 35, 39]] # Certain specific detectors

print(f"Features: {x_columns}")
print(f"Labels: {y_columns}")
print("Features Length :", len(x_columns))
print("Labels Length :", len(y_columns))

y_scaler = preprocessing.StandardScaler()
data[y_columns] = y_scaler.fit_transform(data[y_columns])

x_scaler = preprocessing.StandardScaler()
data[x_columns] = x_scaler.fit_transform(data[x_columns])

IN_FEATURES = len(x_columns)
OUT_FEATURES = len(y_columns)


## Create an individual loss for each label
criterion = TorchLossWrapper(torch.nn.MSELoss(), name="fetal_sat")

## Single Losses
# criterion = TorchLossWrapper(nn.MSELoss(), name="fetal_sat")
# criterion = TorchLossWrapper(nn.CrossEntropyLoss(), name='fetal_sat')


set_seed(40)

## Validation Methods
validation_method = ColumnBasedRandomSplit("Maternal Wall Thickness", 0.8, seed=40)
# all_depths = data["Maternal Wall Thickness"].unique()
# all_depths.sort()
# validation_method = HoldOneOut("Maternal Wall Thickness", all_depths[8:12])  # Center value

dataloader_gen = DataLoaderGenerator(data, x_columns, y_columns, 32, {"shuffle": True})
model = PerceptronBD([IN_FEATURES, 80, 40, OUT_FEATURES])  # Perceptron Model - Change Here
# model = SplitChannelCNN(IN_FEATURES, 3, [9, 18], [9, 7], [20, 10, OUT_FEATURES])    # CNN Model - Change Here

print("Model Summary")
torchinfo.summary(model, input_size=(32, IN_FEATURES))  # Assuming a batch size of 32

trainer = ModelTrainer(model, dataloader_gen, validation_method, criterion, verbose=True)
trainer.set_optimizer(Adam, {"lr": 1e-3, "weight_decay": 1e-4})
# trainer.set_batch_size(128)
trainer.set_batch_size(512)
trainer.run(41)  # Set Epochs Here
print("Training Done")
print("Error Table")
criterion.print_table()

# ## Save Model Code
model_name = "random_split_3wv_60_features_alt"
print(f"Saving Model & Scalers as {model_name}")
torch.save(model.state_dict(), str(PROJECT_BASE_PATH / "models" / f"{model_name}.pt"))
# Save the Scalers for Later Use
joblib.dump(x_scaler, PROJECT_BASE_PATH / "models" / f"{model_name}_xscaler")
joblib.dump(y_scaler, PROJECT_BASE_PATH / "models" / f"{model_name}_yscaler")
