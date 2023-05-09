"""
This file generates SNR from a set of simulation files and stores it into 
a pickle file
"""
import os
from glob import glob
from pathlib import Path
from pandas import read_pickle
import numpy as np
import pandas as pd
from inverse_modelling_tfo.tools.name_decoder import decode_extended_filename
from inverse_modelling_tfo.tools.dataframe_handling import generate_sdd_column_

def get_photon_count_snr(file_path: Path):
    """
    Convert a Pickle containing raw photon partial paths into detector intensity for any given set of absorption co-efficients.
    Uses Beer-Lambert law directly on each detected photon and adds up the intensities.

    The file is stored in the currect working directory with the name <CWD/output_file> 

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    """
    simulation_data = read_pickle(file_path)

    # Convert X,Y, Z co-ordinates to SDD
    generate_sdd_column_(simulation_data)
    simulation_data = simulation_data['SDD']    # Only keep SDD

    # SUM Path
    simulation_data = simulation_data.value_counts()
    simulation_data.name = "SNR"
    simulation_data = simulation_data.to_frame().reset_index()
    simulation_data.rename(columns={'index' : 'SDD'}, inplace=True)
    return simulation_data


if __name__ == '__main__':
    raw_data_path = Path('/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector')
    output_file = os.getcwd() + os.sep + 'snr1.pkl'
    print(f'saving as {output_file}')

    # Get all the simulation files
    all_files = glob(str(raw_data_path.joinpath('*.pkl')))

    combined_df = None
    # Process each file
    # Note: For these RAW files, the saturation and the state parts of the name do not mean anything
    for file in all_files:
        # Get simulation settings using file name
        maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(
            file)
        snr_df = get_photon_count_snr(file)
        num_rows = len(snr_df)
        snr_df['Wave Int'] = wave_int * np.ones((num_rows, 1))
        snr_df['Uterus Thickness'] = uterus_thickness * \
            np.ones((num_rows, 1))
        snr_df['Maternal Wall Thickness'] = maternal_wall_thickness * \
            np.ones((num_rows, 1))
            
        # Add new data to the combined DF
        if combined_df is None:
            combined_df = snr_df
        else:
            combined_df = pd.concat(
                [combined_df, snr_df], axis=0, ignore_index=True)

    combined_df.to_pickle(output_file)
