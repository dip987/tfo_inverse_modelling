"""
Helper functions to normalize simulation intensity values depending on the configuration.
"""
from math import pi
from pandas import DataFrame


CONSTANT_DETECTOR_COUNT = 20
EQUIDISTANCE_DETECTOR_COUNT = [11, 16, 22, 27, 32, 38, 43,
                               48, 53, 59, 64, 69, 75, 80,
                               85, 90, 96, 101, 106, 111]
EQUIDISTANCE_DETECTOR_PHOTON_COUNT = 1e9
COUNSTANT_DETECTOR_COUNT_PHOTON_COUNT = 1e8
SIMULATION_DETECTOR_RADIUS = 2
SIMULATION_UNITINMM = 1.0



def equidistance_detector_normalization(data: DataFrame) -> None:
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
    """
    sdd = data['SDD'].unique()
    sdd.sort()
    sdd_to_detector_count_map = {
        dist: count for dist, count in zip(sdd, EQUIDISTANCE_DETECTOR_COUNT)}
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
