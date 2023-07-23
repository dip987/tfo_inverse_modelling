
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


def get_photon_count_snr(file_path: Path, fetal_layer_number: int = 4):
    """Get photon count on each detector. Caluclates both Total and Fetal Count.
    Creates a DataFrame with 3 columns: SDD, SNR, Fetal SNR

    Args:
        file_path (Path): Simulation Raw Data 
        fetal_layer_number (int, optional): Fetal Layer between 1 to n. Defaults to 4.
    """
    simulation_data = read_pickle(file_path)

    # Convert X,Y, Z co-ordinates to SDD
    generate_sdd_column_(simulation_data)
    simulation_data["isFetal"] = simulation_data[f'L{fetal_layer_number} ppath'] > 0.0

    
    total_count = simulation_data['SDD'].value_counts()
    total_count.name = 'SNR'
    total_count = total_count.to_frame().reset_index()
    total_count.rename(columns={'index': 'SDD'}, inplace=True)
    
    fetal_count = simulation_data.groupby("SDD")['isFetal'].sum()
    fetal_count.name = 'Fetal SNR'
    fetal_count = fetal_count.to_frame().reset_index()
    
    
    combined_count = pd.merge(total_count, fetal_count, on='SDD')
    return combined_count


if __name__ == '__main__':
    raw_data_path = Path(
        '/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector')
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
