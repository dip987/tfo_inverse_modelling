"""
Helper functions to normalize simulation intensity values depending on the configuration.
"""
from pathlib import Path
from typing import Optional
import json
from math import pi
from pandas import DataFrame
from numpy import ndarray

CONSTANT_DETECTOR_COUNT = 20
EQUIDISTANCE_DETECTOR_COUNT = [11, 16, 22, 27, 32, 38, 43,
                               48, 53, 59, 64, 69, 75, 80,
                               85, 90, 96, 101, 106, 111]
EQUIDISTANCE_DETECTOR_PHOTON_COUNT = 1e9
COUNSTANT_DETECTOR_COUNT_PHOTON_COUNT = 1e8
SIMULATION_DETECTOR_RADIUS = 2
SIMULATION_UNITINMM = 1.0



def equidistance_detector_normalization(data: DataFrame, sdd: Optional[ndarray]=None) -> None:
    """Normalize Intensity data from the equidistance detector type
    of simulation. In this setup, the distance between the detectors
    along the radial ring is always equal. Which in turn leads to 
    larger number of detectors for far away rings. 
    
    Normalization dividers includes:
        1. Photon Count
        2. Detector Count
        3. Detector Radius (in mm)

    Args:
        data (DataFrame): Simulation data. The data must include an
        'Intensity' and an 'SDD' column. The 'Intensity' column will
        be modified.
        sdd (Optional[ndarray]): Pass the sorted SDD list for faster execusion.
        You can also choose to pass None, in which case, this code will calculate it
    """
    if sdd is None:
        sdd = data['SDD'].unique()
        sdd.sort()
    sdd_to_detector_count_map = {dist: count for dist, count in zip(sdd, EQUIDISTANCE_DETECTOR_COUNT)}
    data['Intensity'] /= EQUIDISTANCE_DETECTOR_PHOTON_COUNT
    data['Intensity'] /= data['SDD'].map(sdd_to_detector_count_map)
    data['Intensity'] /= pi * SIMULATION_DETECTOR_RADIUS ** 2

def constant_detector_count_normalization(data: DataFrame) -> None:
    """Normalize Intensity data from the constant detector count type
    of simulation. In this setup, detector count in each radial ring is
    always equal.
    
    Normalization dividers includes:
        1. Photon Count
        2. Detector Count
        3. Detector Radius (in mm)

    Args:
        data (DataFrame): Simulation data. The data must include an
        'Intensity' and an 'SDD' column. The 'Intensity' column will
        be modified.
    """
    data['Intensity'] /= COUNSTANT_DETECTOR_COUNT_PHOTON_COUNT
    data['Intensity'] /= CONSTANT_DETECTOR_COUNT
    data['Intensity'] /= pi * SIMULATION_DETECTOR_RADIUS ** 2

def config_based_normalization(data: DataFrame, config_path: Path) -> None:
    """
    Normalize intensity data based on the configuration file.
    
    Normalization dividers includes:
        1. Photon Count
        2. Detector Count
        3. Detector Radius (in mm)
    
    Args:
        data (DataFrame): Simulation data. The data must include an 'Intensity' and an 'SDD' column. The 'Intensity'
        column will be modified.
        config_path (Path): Path to the configuration file. (A JSON containing)
    """
    config = {}
    with open(config_path, 'r') as file:
        config = json.load(file)
    detector_count = config['n_per_det']
    photon_count = config['total_photons_simulated']
    detector_radius = config['det_rad']
    
    sdd = data['SDD'].unique()
    sdd.sort()
    sdd_to_detector_count_map = {dist: count for dist, count in zip(sdd, detector_count)}
    
    # Normalize
    data['Intensity'] /= photon_count
    data['Intensity'] /= data['SDD'].map(sdd_to_detector_count_map)
    data['Intensity'] /= pi * detector_radius ** 2

        