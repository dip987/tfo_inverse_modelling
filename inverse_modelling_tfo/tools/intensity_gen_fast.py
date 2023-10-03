"""
Calculate intensity from RAW simulation files with optimized and parallel methods
"""
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import pandas as pd
from numba import njit

from inverse_modelling_tfo.tools.dataframe_handling import generate_sdd_column_

LAYER_COUNT = 4


# class FastDataGen:
#     def __init__(self, file_path: Path, base_mu_map: np.ndarray, var1_index: int, var2_index: int,
#                  var1_values: np.ndarray, var2_values: np.ndarray):
#         raw_data = FastDataGen._load_data(file_path)
#         self.data_groups = raw_data.groupby("SDD")
#
#         # Find Relevant indices
#         exclude_var2 = list(range(LAYER_COUNT))
#         exclude_var2.pop(var2_index)
#
#         exclude_both_var = list(range(LAYER_COUNT))
#         exclude_both_var.pop(var2_values)
#         exclude_both_var.pop(var1_values)
#
#         self.data_table_static = self.
#
#
#
#
#     @staticmethod
#     def _load_data(file_path: Path) -> pd.DataFrame:
#         simulation_data = pd.read_pickle(file_path)
#         generate_sdd_column_(simulation_data)
#         simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)
#         return simulation_data
#
#     @staticmethod
#     @njit(parallel=True)
#     def _calculate():


# @njit(parallel=True)
# def exp_sum_mem(x: np.ndarray, count: int):
#     all_res = np.zeros((count, x.shape[0]))
#     passed1 = x.copy()
#     passed1[:, :3] = np.exp(-passed1[:, :3])
#     for i in prange(count):
#         e1 = passed1.copy()
#         e1[:, 3] = np.exp(-e1[:, 3])
#         all_res[i] = np.sum(e1, axis=1).flatten()
#     return all_res

@njit(parallel=True, nogil=True)
def intensity_from_raw_fast(simulation_data: pd.DataFrame, mu_map: Dict[int, float],
                            unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Convert a Pickle containing raw photon partial paths into detector intensity for any given set of absorption co-efficients.
    Uses Beer-Lambert law directly on each detected photon and adds up the intensities.

    The file is stored in the current working directory with the name <CWD/output_file>

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    """
    # Convert X,Y, Z co-ordinates to SDD
    generate_sdd_column_(simulation_data)
    simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)

    mu_a_vector = np.array(list(mu_map.values())).reshape(1, -1)
    layer_count = mu_a_vector.shape[1]
    simulation_data.iloc[:, :layer_count] = np.exp(-simulation_data.iloc[:, :layer_count].to_numpy() * mu_a_vector)

    # This line either takes the sum of all photons hitting a certain detector
    simulation_data = simulation_data.groupby(['SDD'])['Intensity'].sum()
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()
    return simulation_data


def intensity_from_raw(simulation_data: pd.DataFrame, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
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
    data = pd.read_pickle(raw_data_path)
    mu_map_base1 = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm

    tic = perf_counter()
    for i in range(10):
        intensity_from_raw_fast(data, mu_map_base1)
    toc = perf_counter()
    print(f"Time Take {toc - tic}")
