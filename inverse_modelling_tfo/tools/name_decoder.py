"""
Deals with decoding filenames to generate tissue simulation model parameters.
"""
from os import sep
from typing import Tuple

def decode_extended_filename(file_name: str) -> Tuple[int, int, int]:
    """Decode a name string and generate simulation model params.

    Args:
        file_name (str): full file name

    Returns:
        Tuple[int, int, int]: Maternal Wall Thickness, Uterus Thickness, Wave Int 
    """
    base_file_names = file_name.split(sep)[-1]
    base_file_names_without_extension = base_file_names[:-4]
    name_components = base_file_names_without_extension.split('_')
    maternal_wall_thickness = int(name_components[-3])
    uterus_thickness = int(name_components[-1])
    wave_int = int(name_components[3])
    return maternal_wall_thickness, uterus_thickness, wave_int
