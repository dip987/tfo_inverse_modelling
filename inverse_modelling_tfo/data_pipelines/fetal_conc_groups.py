"""
Generate/load dictionaries containing fetal concentration groups. All fetal concentration values within the same group
(saved here as a dictionary with a group ID) will be permuated/combinated to generate AC features. All the conc. values
within one group are assumed to be viable pulsation points for the same patient. 
"""

import json
from typing import Dict
from pathlib import Path


def generate_grouping_from_config(config_path: Path, rounding_points: int = 2) -> Dict[float, int]:
    """
    Generate a dictionary containing fetal concentration groups from a configuration file. The configuration file should
    contain a "fconc_centers" key that maps fetal concentration values to group IDs. For an example of the configuration
    files, go to tfo_sim/data/compiled_intensity/dan_iccps_pencil.json. These files should be auto-generated when
    sweeping over mu_a values and saved with the same name as the compiled intensity data file but with a .json.
    :param config_path: Path to the configuration file
    :param rounding_points: Number of decimal points to round the fetal concentration values to. (Default: 2)
    :return: A dictionary containing fetal concentration groups to their mapping IDs
    """
    # Sanity Check
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with config_path.open("r") as file:
        config = json.load(file)

    grouping_map = config["fconc_centers"]  # The key will be strings, needs to be converted to float
    grouping_map = {round(float(k), rounding_points): v for k, v in grouping_map.items()}
    return grouping_map


dan_iccps_gauss2 = {
    10.45: 0,
    11.00: 0,
    11.40: 0,
    11.55: 1,
    12.00: 1,
    12.35: 1,
    12.60: 2,
    13.00: 2,
    13.30: 2,
    13.65: 3,
    14.00: 3,
    14.25: 3,
    14.70: 4,
    15.00: 4,
    15.75: 4,
}

dan_iccps_pencil1 = {
    10.45: 0,
    10.88: 0,
    11.0: 0,
    11.31: 1,
    11.45: 1,
    11.55: 1,
    11.75: 2,
    11.91: 2,
    12.03: 2,
    12.18: 3,
    12.36: 3,
    12.5: 3,
    12.61: 4,
    12.82: 4,
    12.98: 4,
    13.04: 5,
    13.27: 5,
    13.46: 5,
    13.47: 6,
    13.73: 6,
    13.9: 6,
    13.94: 7,
    14.18: 7,
    14.34: 7,
    14.41: 8,
    14.64: 8,
    14.77: 8,
    14.89: 9,
    15.09: 9,
    15.2: 9,
    15.37: 10,
    15.55: 10,
    15.85: 10,
    16.0: 11,
    16.32: 11,
    16.8: 11,
}
