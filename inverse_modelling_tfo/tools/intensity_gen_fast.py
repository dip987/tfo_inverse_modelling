"""
Calculate intensity from RAW simulation photon path files with optimized and parallel methods
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from numba import njit, prange, jit

from inverse_modelling_tfo.tools.dataframe_handling import generate_sdd_column_

SDD_list = np.array([10, 14, 19, 23, 28, 32, 37, 41, 46, 50, 55, 59, 64, 68, 73, 77, 82, 86, 91, 95])


class NoOpDecorator:
    """
    A unity decorator - does nothing - replacement for jit for testing purposes
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# JIT decorator for parallelization and releasing GIL
fast = njit(parallel=True, nogil=True)
# fast = NoOpDecorator


class FastDataGen:
    """
    A Fast JIT based implementation of my intensity data generator. The params are following

    file_path: Path variable
    base_mu_map: numpy array with the relevant mu_a in the same order as the RAW data layers
    var1_index: (0 start) layer index of the first mu_a that changes
    var2_index: (0 start) layer index of the second mu_a that changes
    var1_values: options for the first mu_a as a numpy array
    var2_values: options for the second mu_a as a numpy array
    """

    def __init__(
        self,
        file_path: Path,
        base_mu_map: np.ndarray,
        var1_index: int,
        var2_index: int,
        var1_values: np.ndarray,
        var2_values: np.ndarray,
    ):
        sim_data = FastDataGen._load_data(file_path)

        # Create all required arrays
        self.sdd_one_hot = _create_one_hot(sim_data["SDD"].to_numpy())

        # Adapt mu_a to become a numpy vector
        self.base_mu_a = base_mu_map
        # Find Relevant indices
        exclude_both_var = list(range(len(base_mu_map)))
        exclude_both_var.pop(var2_index)
        exclude_both_var.pop(var1_index)

        # Reshape each array to be vertical columns (Because for some reason Pandas is really stupid)
        self.fixed_pathlength = sim_data.iloc[:, exclude_both_var].to_numpy().reshape(-1, len(exclude_both_var))
        self.var1_pathlengths = sim_data.iloc[:, var1_index].to_numpy().reshape(-1, 1)
        self.var2_pathlengths = sim_data.iloc[:, var2_index].to_numpy().reshape(-1, 1)
        self.fixed_mu_a = base_mu_map[exclude_both_var]
        self.var1_mu_a_options = var1_values
        self.var2_mu_a_options = var2_values

    def run(self):
        return _find_intensity(
            self.fixed_pathlength,
            self.var1_pathlengths,
            self.var2_pathlengths,
            self.sdd_one_hot,
            self.fixed_mu_a,
            self.var1_mu_a_options,
            self.var2_mu_a_options,
        )

    @staticmethod
    def _load_data(file_path: Path) -> pd.DataFrame:
        simulation_data = pd.read_pickle(file_path)
        generate_sdd_column_(simulation_data)
        simulation_data.drop(["X", "Y", "Z"], axis=1, inplace=True)
        return simulation_data


@fast
def _create_one_hot(sdd_array: np.ndarray) -> np.ndarray:
    """Create a one-hot encoding to replace the SDD column in simulation data.
    The classes in the one-hot are based on [SDD_list]

    Args:
        sdd_array (np.ndarray): The SDD column of the Simulation data interpreted as a contiguous numpy array

    Returns:
        np.ndarray: One hot encoding
    """
    sdd_one_hot = np.full((len(SDD_list), sdd_array.shape[0]), 0.0)
    for i in prange(len(SDD_list)):  # pylint: disable=not-an-iterable
        sdd_one_hot[i, :] = sdd_array == SDD_list[i]
    return sdd_one_hot


@fast
def _find_intensity(
    fixed_pathlengths: np.ndarray,
    var1_pathlengths: np.ndarray,
    var2_pathlengths: np.ndarray,
    sdd_one_hot: np.ndarray,
    fixed_mu_a: np.ndarray,
    var1_mu_a_options: np.ndarray,
    var2_mu_a_options: np.ndarray,
) -> np.ndarray:
    """Calculate intensity using the given data in a fast manner
    # TODO: Create documentation for this process at some point

    Args:
        fixed_pathlengths (np.array): Pathlength columns corresponding to the layers that stay static
        var1_pathlengths (np.ndarray): Variable pathlength column 1 (outer loop)
        var2_pathlengths (np.ndarray): Variable pathlength column 2 (inner loop)
        sdd_one_hot (np.array): One-hot SDD encoding for each row
        fixed_mu_a (np.ndarray): 1D fixed mu_a corresponding to fixed_pathlength (column dimensions should match)
        var1_mu_a_options (np.ndarray): all possible mu_a options for the first variable mu_a (mm-1)
        var2_mu_a_options (np.ndarray): all possible mu_a options for the second variable mu_a (mm-1)

    Returns:
        np.ndarray: A flattened numpy array containing the intensity at each SDD for every combination of var1 and var2
        mu_a's (where var1 is the outer loop and var2 is the inner loop)
    """
    # Store the final result from all combinations as one long vector
    final_result = np.zeros((len(SDD_list) * len(var1_mu_a_options) * len(var2_mu_a_options),))
    row_count = len(fixed_pathlengths)
    # Numba requires that during any multiplication all the vectors have the exact same size
    mu_a_fixed_tiled = _mu_tiler(fixed_mu_a, row_count)  # Tiled to match fixed_pathlengths

    # Intensity will ultimately be exp( sum ( - pathlength * mu ) )
    # The sum terms can be Broken Up into 3 parts : Fixed (Doesn't change for a single tissue model), Var1 and Var2

    # Calculate the fixed(Static) term once and use it for each iteration for the same model

    # fixed_sum_term = np.sum(- fixed_pathlengths * mu_a_fixed_tiled, axis=1).reshape(-1, 1)

    fixed_sum_term_all = np.exp(-fixed_pathlengths * mu_a_fixed_tiled)
    fixed_sum_term = np.ones((row_count, 1))
    for fixed_column_index in prange(fixed_sum_term_all.shape[1]):  # pylint: disable=not-an-iterable
        fixed_sum_term *= fixed_sum_term_all[:, fixed_column_index].copy().reshape(-1, 1)

    for i in prange(len(var1_mu_a_options)):  # pylint: disable=not-an-iterable
        var1_mu_a_tiled = _mu_tiler(np.array(var1_mu_a_options[i]), row_count)  # Tiled to match var1_pathlengths
        # var1_mu_a_tiled = np.tile(var1_mu_a_options[i], (row_count, 1))  # Tiled to match var1_pathlengths
        # Var1 + fixed term will stay the same for all iterations involving this specific value of var1_mu_a
        # var1_sum = np.add(- var1_pathlengths * var1_mu_a_tiled, fixed_sum_term)

        var1_sum = np.exp(-var1_pathlengths * var1_mu_a_tiled) * fixed_sum_term

        for j in prange(len(var2_mu_a_options)):  # pylint: disable=not-an-iterable
            var2_mu_a_tiled = _mu_tiler(np.array(var2_mu_a_options[j]), row_count)  # Tiled to match var1_pathlengths
            # var2_mu_a_tiled = np.tile(var2_mu_a_options[j], (row_count, 1))  # Tiled to match var2_pathlengths
            # intensity_column = np.exp(- var2_pathlengths * var2_mu_a_tiled + var1_sum)

            intensity_column = np.exp(-var2_pathlengths * var2_mu_a_tiled) * var1_sum

            # Use a one-hot encoded 2D matrix to sort the sums for each detector
            per_sdd_intensity = _calculate_per_sdd_intensity(intensity_column, sdd_one_hot)
            # per_sdd_intensity = _calculate_per_sdd_intensity(intensity_column, sdd_one_hot)

            # Store result and update pointer
            final_result_pointer = (i * len(var2_mu_a_options) + j) * per_sdd_intensity.shape[0]
            final_result[final_result_pointer : final_result_pointer + per_sdd_intensity.shape[0]] = per_sdd_intensity

    return final_result


def _find_intensity_column(
    fixed_pathlengths: np.ndarray,
    var1_pathlengths: np.ndarray,
    var2_pathlengths: np.ndarray,
    fixed_mu_a: np.ndarray,
    var1_mu_a: float,
    var2_mu_a: float,
):
    row_count = len(fixed_pathlengths)
    mu_a_fixed_tiled = _mu_tiler(fixed_mu_a, row_count)  # Tiled to match fixed_pathlengths
    fixed_sum_term_all = -fixed_pathlengths * mu_a_fixed_tiled
    # fixed_sum_term_all = np.exp(- fixed_pathlengths * mu_a_fixed_tiled)
    fixed_sum_term = np.zeros((row_count, 1))
    for fixed_column_index in prange(fixed_sum_term_all.shape[1]):  # pylint: disable=not-an-iterable
        fixed_sum_term += fixed_sum_term_all[:, fixed_column_index].reshape(-1, 1)
    var1_mu_a_tiled = _mu_tiler(np.array(var1_mu_a), row_count)  # Tiled to match var1_pathlengths
    var1_sum = -var1_pathlengths * var1_mu_a_tiled + fixed_sum_term
    var2_mu_a_tiled = _mu_tiler(np.array(var2_mu_a), row_count)
    intensity_column = np.exp(-var2_pathlengths * var2_mu_a_tiled + var1_sum)
    return intensity_column


@njit
def _mu_tiler(mu_a: np.ndarray, vertical_repeats: int) -> np.ndarray:
    """
    Vertically tiles the given 1D mu_a map
    """
    return mu_a.T.repeat(vertical_repeats).reshape(-1, vertical_repeats).T


@fast
def _calculate_per_sdd_intensity(intensity: np.ndarray, sdd_one_hot: np.ndarray) -> np.ndarray:
    """
    Given an intensity column and one-hot sdd's, calcualtes the intensity at each SDD and packs them in a single
    flattened numpy array (The ordering is according to [SDD_list])
    """
    result = np.zeros((sdd_one_hot.shape[0],), dtype="float64")
    for i in prange(sdd_one_hot.shape[0]):  # pylint: disable=not-an-iterable
        result[i] = np.dot(intensity.flatten(), sdd_one_hot[i, :])
    return result


if __name__ == "__main__":
    raw_data_path = Path(
        "/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector/fa_1_wv_1_sa_0.1_ns_1_ms_10_ut_5.pkl"
    )
    mu_map_base1 = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm

    a = FastDataGen(
        raw_data_path,
        np.array([0.0091, 0.0158, 0.0125, 0.013]),
        0,
        3,
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        np.array([0.1, 0.3]),
    )
    tic = perf_counter()
    res = a.run()
    toc = perf_counter()
    print(f"Time Take {toc - tic}")
