"""
Generate a set of simulation data based on a range of independent saturation values and 
concentration for the mom and fetus. The data is generated using RAW simulation files by only
changing the absorption co-efficient as a *function of Saturation* for both layers.

The absorption co-efficient is calcualted with the assumption that there is only Hemoglobain in the
Maternal and Fetal Layer

Running this script will save the data in a .pickle file which can be used for further analysis.
"""
from pathlib import Path
import os
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from inverse_modelling_tfo.tools.intensity_datagen import intensity_from_raw
from inverse_modelling_tfo.tools.name_decoder import decode_extended_filename


def get_mu_a(saturation: float, concentration: float, wave_int: int) -> float:
    """Calculate the absorption co-efficient of the maternal layer using the given parameters

    Args:
        saturation (float): Maternal Layer Saturation [0, 1.0]
        concentration (float): Hb concentration for the maternal layer in g/dL
        wave_int (int): wavelength of light. Set to 1 for 735nm and 2 for 850nm

    Returns:
        float: absorption co-efficient
    """
    wave_index = wave_int - 1
    # Constants
    # 735nm, 850nm, 810nm
    # Values taken from Takatani(1987), https://omlc.org/spectra/hemoglobin/takatani.html
    # All values in cm-1/M
    E_HB = [412, 1058, 880]
    E_HBO2 = [1464, 820, 888]

    # Convert concentration from g/dL to Moles/liter
    # per dL -> per L : times 10; g -> M : divide by grams per Mole
    # Assume HB and HBO2 have similar molar mass
    concentration = concentration * 10 / 64500  # in M/L
    # Notes: molar conc. is usually around 150/64500 M/L for regular human blood

    # Use mu_a formula : mu_a = 2.303 * E * Molar Concentration
    mu_a = 2.303 * concentration * \
        (saturation * E_HB[wave_index] + (1 - saturation) * E_HBO2)  # in cm-1

    mu_a = mu_a / 10  # Conversion to mm-1
    return mu_a

# Generate Intensity Values


if __name__ == '__main__':
    raw_data_path = Path(
        '/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector')
    mu_map_base1 = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm
    mu_map_base2 = {1: 0.0087, 2: 0.0991, 3: 0.042, 4: 0.012}   # 850nm
    # Ranges taken by literally googling around
    c_m_list = np.linspace(12, 16, num=10, endpoint=True)
    c_f_list = np.linspace(11, 17, num=10, endpoint=True)
    # Regular saturation ranges
    s_m_list = np.linspace(0.9, 1.0, num=10, endpoint=True)
    s_f_list = np.linspace(0.1, 0.6, num=10, endpoint=True)

    output_file = os.getcwd() + os.sep + 's_based_intensity.pkl'
    print(f'saving as {output_file}')

    # Get all the simulation files
    all_files = glob(str(raw_data_path.joinpath('*.pkl')))

    total_count = len(c_m_list) * len(c_f_list) * len(s_m_list) * len(s_f_list) * len(all_files)
    print(f'Total Simulation points {total_count}')
    
    combined_df = None
    # Process each file
    # Note: For these RAW files, the saturation and the state parts of the name do not mean anything
    with tqdm(total=total_count) as pbar:
        for file in all_files:
            # Get simulation settings using file name
            maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(
                file)

            # Get intensity
            mu_map_active = mu_map_base1.copy() if wave_int == 1 else mu_map_base2.copy()
            # Try all possible combos of maternal and fetal mu_a for each file
            for c_m in c_m_list:
                for s_m in s_m_list:
                    mu_map_active[1] = get_mu_a(s_m, c_m, wave_int)
                    for c_f in c_f_list:
                        for s_f in s_m_list:
                            mu_map_active[4] = get_mu_a(s_f, c_f, wave_int)
                            intensity_df = intensity_from_raw(file, mu_map_active)
                            num_rows = len(intensity_df)
                            intensity_df['Wave Int'] = wave_int * np.ones((num_rows, 1))
                            intensity_df['Uterus Thickness'] = uterus_thickness * \
                                np.ones((num_rows, 1))
                            intensity_df['Maternal Wall Thickness'] = maternal_wall_thickness * \
                                np.ones((num_rows, 1))
                            intensity_df['Maternal Hb Concentration'] = c_m * np.ones((num_rows, 1))
                            intensity_df['Maternal Saturation'] = s_m * np.ones((num_rows, 1))
                            intensity_df['Fetal Hb Concentration'] = c_f * np.ones((num_rows, 1))
                            intensity_df['Fetal Saturation'] = s_f * np.ones((num_rows, 1))

                            # Add new data to the combined DF
                            if combined_df is None:
                                combined_df = intensity_df
                            else:
                                combined_df = pd.concat(
                                    [combined_df, intensity_df], axis=0, ignore_index=True)
                            tqdm.update(1)

        combined_df.to_pickle(output_file)
