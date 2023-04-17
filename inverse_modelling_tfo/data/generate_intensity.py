import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd


def intensity_from_distribution(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Convert a CSV containing photon path densities into detector intensity for any given set of absorption co-efficients.
    Uses the method described in Fredriksson et al. 2012. For a detailed description of each step in the code please
    look at the notebook named "data_format_exploration.ipynb".

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    """

    simulation_data = pd.read_csv(file_path,
                                  dtype={'Ppath Medium': np.int32, 'Deepest Layer': np.int32, 'Count': np.int32})

    # Convert X,Y, Z co-ordinates to SDD
    varying_coordinate = 'X' if len(simulation_data['X'].unique()) > 1 else 'Y'
    fixed_coordinate = 'X' if varying_coordinate == 'Y' else 'Y'
    source_coordinate = simulation_data[fixed_coordinate][0]
    simulation_data['SDD'] = (simulation_data[varying_coordinate] - source_coordinate).astype(
        np.int32) * unitinmm  # in mm
    simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)

    # Normalization
    normalization_factor = simulation_data.groupby(['Deepest Layer', 'SDD'])['Count'].transform('sum')
    normalization_factor_grouped = simulation_data.groupby(['Deepest Layer', 'SDD'])['Count'].sum()
    simulation_data['Count'] = simulation_data['Count'] / normalization_factor

    # Weighted Path Length
    simulation_data['Weighted Ppath'] = simulation_data['Count'] * np.exp(
        -simulation_data['Bin Center'] * unitinmm * [mu_map[medium] for medium in simulation_data['Ppath Medium']])

    # Taking the area under the distribution
    simulation_data = simulation_data.groupby(['Ppath Medium', 'Deepest Layer', 'SDD'], sort=False, as_index=False)[
        'Weighted Ppath'].sum()

    # Multiplying distribution from each layer
    simulation_data = simulation_data.groupby(['Deepest Layer', 'SDD'], sort=False)['Weighted Ppath'].prod()

    # Initial Intensity I_0
    simulation_data = simulation_data * normalization_factor_grouped

    # Sum up
    simulation_data = simulation_data.groupby(['SDD']).sum()

    # Rename and create df
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()

    return simulation_data


def intensity_from_raw(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Convert a CSV containing raw photon partial paths into detector intensity for any given set of absorption co-efficients.
    Uses Beer-Lambert law directly on each detected photon and adds up the intensities.

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    The intensity is summed across all detectors in a single SDD(Each detector ring). When comparing between different setups,
    make sure to divide the intensity by the number of detectors. 
    """
    simulation_data = pd.read_pickle(file_path)

    # Convert X,Y, Z co-ordinates to SDD
    varying_coordinate = 'X' if len(simulation_data['X'].unique()) > 1 else 'Y'
    fixed_coordinate = 'X' if varying_coordinate == 'Y' else 'Y'
    source_coordinate = simulation_data[fixed_coordinate][0]
    simulation_data['SDD'] = (simulation_data[varying_coordinate] - source_coordinate).astype(np.int32)  # in mm
    simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)

    available_layers = []
    # Take the exponential
    for layer in mu_map.keys():
        if f'L{layer} ppath' in simulation_data.columns:
            simulation_data[f'L{layer} ppath'] = np.exp(-simulation_data[f'L{layer} ppath'] * unitinmm * mu_map[layer])
            available_layers.append(f'L{layer} ppath')

    # Get Intensity
    simulation_data['Intensity'] = simulation_data[available_layers].prod(axis=1)

    # Sum per detector
    simulation_data = simulation_data.groupby(['SDD'])['Intensity'].sum()

    # Rename and create df
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()

    return simulation_data


if __name__ == '__main__':
    # Usage
    # path = Path(r'C:\Users\sadip\PycharmProjects\tfo_inverse_modelling\data\raw\fa_1_wv_1_sa_0.1_ns_1_ms_5.csv')
    path = Path(r'C:\Users\sadip\PycharmProjects\tfo_inverse_modelling\data\raw\fa_1_wv_1_sa_0.1_ns_1_ms_5.pkl')
    mu = {1: 0.017, 2: 0.0085, 3: 0.016, 4: 0.0125, 5: 0.0157, 6: 0.0175, 7: 0.15058259000000002, 8: 0.0187}
    # intensity_data = intensity_from_distribution(path, mu)
    intensity_data = intensity_from_raw(path, mu)
    print(intensity_data)
