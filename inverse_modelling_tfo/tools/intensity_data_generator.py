"""
This file generates intensity data using RAW type simulation outputs. We can change the absorption co-eff and generate different sets of data. This is meant to train a forward neural network.
"""
from typing import Dict
import numpy as np
from pathlib import Path
from glob import glob
import pandas as pd
import os

raw_data_path = Path('/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps')
mu_map_base1 = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm
mu_map_base2 = {1: 0.0087, 2: 0.0991, 3: 0.042, 4: 0.012}   # 850nm
output_file = 'intensity_averaged_sim_data.pkl'
fetal_mu_a = np.arange(0.05, 0.10, 0.005)
maternal_mu_a = np.arange(0.005, 0.010, 0.0005)



# Generate Intensity Values
def intensity_from_raw(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Convert a Pickle containing raw photon partial paths into detector intensity for any given set of absorption co-efficients.
    Uses Beer-Lambert law directly on each detected photon and adds up the intensities.

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    """
    simulation_data = pd.read_pickle(file_path)

    # Convert X,Y, Z co-ordinates to SDD
    varying_coordinate = 'X' if len(simulation_data['X'].unique()) > 1 else 'Y'
    fixed_coordinate = 'X' if varying_coordinate == 'Y' else 'Y'
    source_coordinate = simulation_data[fixed_coordinate][0]
    simulation_data['SDD'] = (
        simulation_data[varying_coordinate] - source_coordinate).astype(np.int32)  # in mm
    simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)

    available_layers = []
    # Take the exponential
    for layer in mu_map.keys():
        if f'L{layer} ppath' in simulation_data.columns:
            simulation_data[f'L{layer} ppath'] = np.exp(
                -simulation_data[f'L{layer} ppath'] * unitinmm * mu_map[layer])
            available_layers.append(f'L{layer} ppath')

    # Get Intensity
    simulation_data['Intensity'] = simulation_data[available_layers].prod(
        axis=1)

    # TODO: Change to sum later?
    # Mean(instead of sum) per detector
    simulation_data = simulation_data.groupby(['SDD'])['Intensity'].sum()

    # Rename and create df
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()

    return simulation_data


# Get all the simulation files
all_files = glob(str(raw_data_path.joinpath('*.pkl')))

combined_df = None
# Process each file
# Note: For these RAW files, the saturation and the state parts of the name do not mean anything
for file in all_files:
    # Get simulation settings using file name
    base_file_names = file.split(os.sep)[-1]
    base_file_names_without_extension = base_file_names[:-4]
    name_components = base_file_names_without_extension.split('_')
    uterus_thickness = int(name_components[-1])
    maternal_wall_thickness = int(name_components[-3])
    wave_int = int(name_components[3])

    # Get intensity
    mu_map = mu_map_base1 if wave_int == 1 else mu_map_base2
    # Try all possible combos of maternal and fetal mu_a for each file
    for f_mu_a in fetal_mu_a:
        for m_mu_a in maternal_mu_a:
            mu_map[1] = m_mu_a  # Change maternal mu a
            mu_map[4] = f_mu_a  # Change fetal mu a
            intensity_df = intensity_from_raw(file, mu_map)
            num_rows = len(intensity_df)
            intensity_df['Wave Int'] = wave_int * np.ones((num_rows, 1))
            intensity_df['Uterus Thickness'] = uterus_thickness * np.ones((num_rows, 1))
            intensity_df['Maternal Wall Thickness'] = maternal_wall_thickness * np.ones((num_rows, 1))
            intensity_df['Maternal Mu_a'] = mu_map[1] * np.ones((num_rows, 1))
            intensity_df['Fetal Mu_a'] = mu_map[4] * np.ones((num_rows, 1))

            # Add new data to the combined DF
            if combined_df is None:
                combined_df = intensity_df
            else:
                combined_df = pd.concat([combined_df, intensity_df], axis=0, ignore_index=True)

combined_df.to_pickle(output_file)