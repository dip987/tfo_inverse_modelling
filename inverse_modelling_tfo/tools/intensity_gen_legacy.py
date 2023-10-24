"""
Generate intensity from RAW photon path length files. (Legacy)
This is an older/slower implementation. Use the functions in intensity_gen_fast.py for faster execution times
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from inverse_modelling_tfo.tools.dataframe_handling import generate_sdd_column_


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

def intensity_column_from_raw(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
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
    return simulation_data
