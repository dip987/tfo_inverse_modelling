"""
This file generates intensity data using RAW type simulation outputs. We can change the absorption 
co-eff and generate different sets of data. This is meant to train a forward neural network.
"""
import os
from glob import glob
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from inverse_modelling_tfo.tools.dataframe_handling import generate_sdd_column_
from inverse_modelling_tfo.tools.name_decoder import decode_extended_filename


# Generate Intensity Values
def intensity_from_raw(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Convert a Pickle containing raw photon partial paths into detector intensity for any given set of absorption co-efficients.
    Uses Beer-Lambert law directly on each detected photon and adds up the intensities.
    
    The file is stored in the currect working directory with the name <CWD/output_file> 

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    """
    simulation_data = pd.read_pickle(file_path)

    # Convert X,Y, Z co-ordinates to SDD
    generate_sdd_column_(simulation_data)
    simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)

    available_layers = []
    # Take the exponential
    for layer in mu_map.keys():
        if f'L{layer} ppath' in simulation_data.columns:
            simulation_data[f'L{layer} ppath'] = np.exp(-simulation_data[f'L{layer} ppath'] * unitinmm * mu_map[layer])
            available_layers.append(f'L{layer} ppath')

    # Get Intensity
    # This creates the intensity of each photon individually
    simulation_data['Intensity'] = simulation_data[available_layers].prod(axis=1)

    # This line either takes the sum of all photons hitting a certain detector
    simulation_data = simulation_data.groupby(['SDD'])['Intensity'].sum()
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()
    return simulation_data


if __name__ == '__main__':
    raw_data_path = Path('/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector')
    mu_map_base1 = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm
    mu_map_base2 = {1: 0.0087, 2: 0.0991, 3: 0.042, 4: 0.012}  # 850nm
    fetal_mu_a = np.arange(0.05, 0.30, 0.04)
    maternal_mu_a = np.arange(0.005, 0.050, 0.004)
    output_file = os.getcwd() + os.sep + 'intensity_summed_sim_data_equidistance_detector.pkl'
    print(f'saving as {output_file}')

    # Get all the simulation files
    all_files = glob(str(raw_data_path.joinpath('*.pkl')))

    combined_df = None
    # Process each file
    # Note: For these RAW files, the saturation and the state parts of the name do not mean anything
    for file in all_files:
        # Get simulation settings using file name
        maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(file)

        # Get intensity
        mu_map_active = mu_map_base1 if wave_int == 1 else mu_map_base2
        # Try all possible combos of maternal and fetal mu_a for each file
        for f_mu_a in fetal_mu_a:
            for m_mu_a in maternal_mu_a:
                mu_map_active[1] = m_mu_a  # Change maternal mu a
                mu_map_active[4] = f_mu_a  # Change fetal mu a
                intensity_df = intensity_from_raw(file, mu_map_active)
                num_rows = len(intensity_df)
                intensity_df['Wave Int'] = wave_int * np.ones((num_rows, 1))
                intensity_df['Uterus Thickness'] = uterus_thickness * np.ones((num_rows, 1))
                intensity_df['Maternal Wall Thickness'] = maternal_wall_thickness * np.ones((num_rows, 1))
                intensity_df['Maternal Mu_a'] = mu_map_active[1] * np.ones((num_rows, 1))
                intensity_df['Fetal Mu_a'] = mu_map_active[4] * np.ones((num_rows, 1))

                # Add new data to the combined DF
                if combined_df is None:
                    combined_df = intensity_df
                else:
                    combined_df = pd.concat([combined_df, intensity_df], axis=0, ignore_index=True)

    combined_df.to_pickle(output_file)
