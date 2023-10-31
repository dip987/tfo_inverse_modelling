"""
Generate a set of simulation data based on a range of independent saturation values and 
concentration for the mom and fetus. The data is generated using RAW simulation files by only
changing the absorption co-efficient as a *function of Saturation* for both layers.

The absorption co-efficient is calcualted with the assumption that there is only Hemoglobain in the
Maternal and Fetal Layer

Running this script will save the data in a .pickle file which can be used for further analysis.
"""
from pathlib import Path
from itertools import product
import os
from glob import glob
import pandas as pd
import numpy as np
from inverse_modelling_tfo.tools.intensity_gen_fast import FastDataGen, SDD_list
from inverse_modelling_tfo.tools.name_decoder import decode_extended_filename


MU_MAP_BASE1 = np.array([0.0091, 0.0158, 0.0125, 0.013])  # 735nm
MU_MAP_BASE2 = np.array([0.0087, 0.0991, 0.042, 0.012])  # 850nm


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
    E_HB = [412., 1058., 880.]
    E_HBO2 = [1464., 820., 888.]

    # Convert concentration from g/dL to Moles/liter
    # per dL -> per L : times 10; g -> M : divide by grams per Mole
    # Assume HB and HBO2 have similar molar mass
    concentration = concentration * 10 / 64500  # in M/L
    # Notes: molar conc. is usually around 150/64500 M/L for regular human blood

    # Use mu_a formula : mu_a = 2.303 * E * Molar Concentration
    mu_a = 2.303 * concentration * (saturation * E_HB[wave_index] + (1 - saturation) * E_HBO2[wave_index])  # in cm-1

    mu_a = mu_a / 10  # Conversion to mm-1
    return mu_a

# Generate Intensity Values
if __name__ == '__main__':
    MODE_APPEND = True
    DEPTH_CUTOFF_LOWER = 2    # Integer - cutoff depth(inclusive)
    DEPTH_CUTOFF_UPPER = 12    # Integer - cutoff depth(inclusive)
    raw_data_path = Path('/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector')

    # Ranges taken by literally googling around
    c_m_list = np.linspace(12.2, 16.2, num=5, endpoint=True)
    c_f_list = np.linspace(0.11, 0.17, num=5, endpoint=True)
    # c_f_list = np.array([0.11])
    # Regular saturation ranges
    s_m_list = np.linspace(0.9, 1.0, num=5, endpoint=True)
    # s_m_list = np.array([0.9])
    s_f_list = np.linspace(0.1, 0.6, num=5, endpoint=True)
    # s_f_list = np.array([0.1])

    # output_file = os.getcwd() + os.sep + 's_based_intensity_low_conc.pkl'
    output_file = os.getcwd() + os.sep + 's_based_intensity_low_conc3.pkl'
    print(f'saving as {output_file}')

    # Get all the simulation files
    all_files = glob(str(raw_data_path.joinpath('*.pkl')))

    # Cutoff
    filtered_by_depth = []
    for file in all_files:
        maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(file)
        if (maternal_wall_thickness >= DEPTH_CUTOFF_LOWER) and (maternal_wall_thickness <= DEPTH_CUTOFF_UPPER):
            filtered_by_depth.append(file)

    all_files = filtered_by_depth

    total_count = len(c_m_list) * len(c_f_list) * len(s_m_list) * len(s_f_list) * len(all_files)
    print(f'Total Simulation points {total_count}')

    combined_df = pd.read_pickle(output_file) if MODE_APPEND else pd.DataFrame()
    # Process each file
    # Note: For these RAW files, the saturation and the state parts of the name do not mean anything

    for file in all_files:
        # Get simulation settings using file name
        maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(file)

        # Get intensity
        base_mu_map = MU_MAP_BASE1.copy() if wave_int == 1 else MU_MAP_BASE2.copy()
        
        # Try all possible combos of maternal and fetal mu_a for each file
        all_sat_con_mom = list(product(s_m_list, c_m_list))
        all_sat_con_fetus = list(product(s_f_list, c_f_list))
        all_mu_a_mom = np.array([get_mu_a(sat, conc, wave_int) for sat, conc in all_sat_con_mom])
        all_mu_a_fetus = np.array([get_mu_a(sat, conc, wave_int) for sat, conc in all_sat_con_fetus])

        # Calculate Intensity - chaning the 0th and 3rd layer index coeffs
        intensity_column = FastDataGen(Path(file), base_mu_map, 0, 3, all_mu_a_mom, all_mu_a_fetus).run()
        # PS: Don't forget to call run on the FastDataGen
        num_rows = len(intensity_column)
        sdd_column = np.tile(SDD_list, num_rows // len(SDD_list))
        wave_int_column = wave_int * np.ones((num_rows, ))
        uterus_thickness_column = uterus_thickness * np.ones((num_rows, ))
        maternal_wall_thickness_column = maternal_wall_thickness * np.ones((num_rows, ))
        fetal_saturation_column = np.tile(np.repeat(np.array([x[0] for x in all_sat_con_fetus]), len(SDD_list)),
                                           len(all_mu_a_mom))
        fetal_concentration_column = np.tile(np.repeat(np.array([x[1] for x in all_sat_con_fetus]), len(SDD_list)),
                                            len(all_mu_a_mom))
        mom_saturation_column = np.repeat(np.array([x[0] for x in all_sat_con_mom]), len(intensity_column) // len(all_sat_con_mom))
        mom_concentration_column = np.repeat(np.array([x[1] for x in all_sat_con_mom]), len(intensity_column) // len(all_sat_con_mom))
        intensity_df = pd.DataFrame({
              'Wave Int': wave_int_column,
              'SDD': sdd_column,
              'Uterus Thickness': uterus_thickness_column,
              'Maternal Wall Thickness': maternal_wall_thickness_column,
              'Maternal Hb Concentration': mom_concentration_column,
              'Maternal Saturation': mom_saturation_column,
              'Fetal Hb Concentration': fetal_concentration_column,
              'Fetal Saturation': fetal_saturation_column, 
              "Intensity": intensity_column
        })
        combined_df = pd.concat([combined_df, intensity_df], axis=0, ignore_index=True)
        # Save Files
        combined_df.to_pickle(output_file)
