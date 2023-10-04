"""
Calculate intensity from RAW simulation files with optimized and parallel methods
"""
from pathlib import Path
from time import perf_counter
from typing import Dict

import numpy as np
import pandas as pd
from numba import njit, prange

from inverse_modelling_tfo.tools.dataframe_handling import generate_sdd_column_

LAYER_COUNT = 4


class FastDataGen:
    def __init__(self, file_path: Path, base_mu_map: np.ndarray, var1_index: int, var2_index: int,
                 var1_values: np.ndarray, var2_values: np.ndarray):
        raw_data = FastDataGen._load_data(file_path)
        self.data_groups = raw_data.groupby("SDD")

        # Find Relevant indices
        exclude_var2 = list(range(LAYER_COUNT))
        exclude_var2.pop(var2_index)

        exclude_both_var = list(range(LAYER_COUNT))
        exclude_both_var.pop(var2_values)
        exclude_both_var.pop(var1_values)

        self.data_table_static = self.




    @staticmethod
    def _load_data(file_path: Path) -> pd.DataFrame:
        simulation_data = pd.read_pickle(file_path)
        generate_sdd_column_(simulation_data)
        simulation_data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)
        return simulation_data

    @staticmethod
    @njit(parallel=True)
    def _calculate():

def _find_intensity(file_path: Path, base_mu_map: np.ndarray, var1_index: int, var2_index: int,
                    var1_values: np.ndarray, var2_values: np.ndarray):
    



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


def intensity_from_raw_fast1(simulation_data: pd.DataFrame, mu_map: Dict[int, float],
                             unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Roughly 3x Speed Improvement over original at doing 10 operations
    """
    mu_a_vector = np.array(list(mu_map.values())).reshape(1, -1)
    mu_a_vector = np.tile(mu_a_vector, (len(simulation_data), 1))
    layer_count = mu_a_vector.shape[1]
    simulation_data["Intensity"] = _calculate_exps(
        simulation_data.iloc[:, :layer_count].to_numpy(), mu_a_vector)

    # This line either takes the sum of all photons hitting a certain detector
    simulation_data = simulation_data.groupby(['SDD'])['Intensity'].sum()
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()
    return simulation_data


@njit(parallel=True, nogil=True)
def _calculate_exps(vect1: np.ndarray, vect2: np.ndarray):
    return np.exp(np.sum(- vect1 * vect2, axis=1))


SDD_list = np.array([10, 14, 19, 23, 28, 32, 37, 41, 46,
                    59, 55, 59, 64, 68, 73, 77, 82, 86, 91, 95])


def intensity_from_raw_fast2(simulation_data: pd.DataFrame, mu_map: Dict[int, float], repeat: int) -> None:
    """
    Vectorized Grouping
    """
    mu_a_vector = np.array(list(mu_map.values())).reshape(1, -1)
    mu_a_vector = np.tile(mu_a_vector, (len(simulation_data), 1))
    layer_count = mu_a_vector.shape[1]

    sdd_one_hot = np.full((len(SDD_list), len(simulation_data)), 0.0)
    for i in range(len(SDD_list)):
        sdd_one_hot[i, :] = simulation_data['SDD'] == SDD_list[i]

    for i in range(repeat):
        _ = _calculate_per_sdd_intensity(simulation_data.iloc[:, :layer_count].to_numpy(), mu_a_vector, sdd_one_hot)


@njit(parallel=True, nogil=True)
def _calculate_per_sdd_intensity(vect1: np.ndarray, vect2: np.ndarray, sdd_one_hot: np.ndarray):
    intensity = _calculate_exps(vect1, vect2)
    result = np.zeros(sdd_one_hot.shape[0])
    for i in prange(sdd_one_hot.shape[0]):
        result[i] = np.dot(intensity, sdd_one_hot[i, :])
    return result

@njit(parallel=True, nogil=True)
def _fixed_sum_term(fixed_columns: np.ndarray, mu_a_tiled: np.ndarray):
    """During calculation of intensity, there will be a fixed sum term coming from the static parts for each model. 
    
    This function pre-calculates it for reuse for all iterations on that model. 
    Args:
        fixed_columns (np.ndarray): Static layer's pathlength columns(L in mm)
        mu_a_tiled (np.ndarray): Corresponidng mu_a for each static layer (mm-1) Tiled to be the same 2D size as 
        fixed_columns (Numba requires these sizes to be the same)
    """
    return np.sum(-fixed_columns * mu_a_tiled, axis=1)

def _col_sum(col1: )





def intensity_from_raw(simulation_data: pd.DataFrame, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    available_layers = []
    # Take the exponential
    for layer in mu_map.keys():
        if f'L{layer} ppath' in simulation_data.columns:
            simulation_data[f'L{layer} ppath'] = np.exp(
                -simulation_data[f'L{layer} ppath'] * unitinmm * mu_map[layer])
            available_layers.append(f'L{layer} ppath')

    # Get Intensity
    # This creates the intensity of each photon individually
    simulation_data['Intensity'] = simulation_data[available_layers].prod(
        axis=1)

    # This line either takes the sum of all photons hitting a certain detector
    simulation_data = simulation_data.groupby(['SDD'])['Intensity'].sum()
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()
    return simulation_data


if __name__ == '__main__':
    raw_data_path = Path(
        "/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector/fa_1_wv_1_sa_0.1_ns_1_ms_10_ut_5.pkl")
    data = pd.read_pickle(raw_data_path)
    # Convert X,Y, Z co-ordinates to SDD
    data['SDD'] = (data['X'] - 100).astype('int32')
    data.drop(['X', 'Y', 'Z'], axis=1, inplace=True)
    mu_map_base1 = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm

    tic = perf_counter()
    intensity_from_raw_fast2(data, mu_map_base1, 10)
    # for i in range(10):
    #     # cats = intensity_from_raw_fast1(data, mu_map_base1)
    #     cats = intensity_from_raw(data, mu_map_base1)
    # print(cats)
    toc = perf_counter()
    print(f"Time Take {toc - tic}")
