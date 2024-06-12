"""
A pipeline to create fetal partial pathlength(ppath L4) bins from RAW simulation data. This is meant to serve as an 
additional physics based loss function for the inverse modelling task.
"""

from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from tfo_sim.misc import generate_sdd_column
from inverse_modelling_tfo.data_pipelines.custom_L4_hist import custom_histogram

RAW_BASE_DIR = Path(r"/home/rraiyan/simulations/tfo_sim/data/dan_iccps_pencil2")  # Path to the RAW simulation data
raw_files = list(RAW_BASE_DIR.glob("*.pkl"))  # RAW files are in pickle format
raw_files = [Path(file) for file in raw_files]

CHOSEN_DETECTOR_INDEX = 2
BIN_COUNT = 10
ROUND_DECIMALS = 5
out_dest = Path(__file__).parent.parent.parent / "data" / "processed_data" / "l4_bins.pkl"

combined_df = pd.DataFrame()

# Add a timer
for raw_file in tqdm(raw_files):
    # Read Data
    data = pd.read_pickle(raw_file)
    sdd_col = generate_sdd_column(data)
    all_sdd = sdd_col.unique()
    all_sdd.sort()
    chosen_sdd = all_sdd[CHOSEN_DETECTOR_INDEX]
    data = data[sdd_col == chosen_sdd]  # Filter data for the chosen detector
    data = data["L4 ppath"]     # Only consider the L4 ppath data

    # Read Config
    config_path = raw_file.with_suffix(".json")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Create Bins
    hist, bin_centers = custom_histogram(data, BIN_COUNT)
    bin_centers = np.round(bin_centers, ROUND_DECIMALS)
    new_row = {str(center): value for center, value in zip(bin_centers, hist)}
    new_row["Wave Int"] = config["wave_int"]
    new_row["Maternal Wall Thickness"] = config["dermis_thickness"]

    # Append to the combined dataframe
    # The bin_centers are always the same - so the column names should also remain the same
    combined_df = pd.concat([combined_df, pd.DataFrame([new_row])], ignore_index=True)

# Save
combined_df.to_pickle(out_dest)

# Create a config file
config_dest = out_dest.with_suffix(".json")
config = {
    "features": [str(center) for center in bin_centers],  
    "bin_count": BIN_COUNT,
    "round_decimals": ROUND_DECIMALS,
    "chosen_detector_index": CHOSEN_DETECTOR_INDEX,
    "Comments": "This file contains the L4 bins for the fetal partial pathlength(ppath) data. The bin centers are not \
        normalized to either photon count or detector count.",
}

with open(config_dest, "w+", encoding="utf-8") as file:
    json.dump(config, file, indent=4)
