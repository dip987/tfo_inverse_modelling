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
from inverse_modelling_tfo.tools.intensity_gen_fast import FastDataGen, SDD_list
from inverse_modelling_tfo.tools.name_decoder import decode_extended_filename
from inverse_modelling_tfo.tools.parameter_generation import MuAGenerator, TMPColumnGenerator

# Default mu_a values for the 4 layers: Maternal Wall, Maternal Uterus, Amniotic Fluid, Fetal Layer
MU_MAP_BASE1 = np.array([0.0091, 0.0158, 0.0125, 0.013])  # 735nm
MU_MAP_BASE2 = np.array([0.0087, 0.0991, 0.042, 0.012])  # 850nm


# Generate Intensity Values
if __name__ == "__main__":
    DEPTH_CUTOFF_LOWER = 6  # Integer - cutoff depth(inclusive)
    DEPTH_CUTOFF_UPPER = 14  # Integer - cutoff depth(inclusive)
    raw_data_path = Path("/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector")

    # Generate all possible mu_a for the given range of saturation and concentration

    # Always appends the data at the end of the current file
    # output_file = os.getcwd() + os.sep + 's_based_intensity_low_conc.pkl'
    output_file = os.getcwd() + os.sep + "s_based_intensity_low_conc6.pkl"
    # TODO: Combine 6 with 5 eventually
    print(f"saving as {output_file}")

    # Get all the simulation files
    all_files = glob(str(raw_data_path.joinpath("*.pkl")))

    # Cutoff
    filtered_by_depth = []
    for file in all_files:
        maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(file)
        if (maternal_wall_thickness >= DEPTH_CUTOFF_LOWER) and (maternal_wall_thickness <= DEPTH_CUTOFF_UPPER):
            filtered_by_depth.append(file)

    all_files = filtered_by_depth

    # Process each file
    # Note: For these RAW files, the saturation and the state parts of the name do not mean anything

    # Create a loading bar
    # with tqdm(total=len(all_files)) as pbar:
    with tqdm(total=len(all_files) * 3) as pbar:  # Running each file 3 times
        for file in all_files:
            # Get simulation settings using file name
            maternal_wall_thickness, uterus_thickness, wave_int = decode_extended_filename(file)

            # Create a MuAGenerator object - Note: The mu_a won't be generated until generate() is called
            mu_a_gen = MuAGenerator((0.9, 1.0), 5, (11, 15), 5, (0.2, 0.6), 5, (11, 15), 5, 0.2, 0.22, wave_int)

            # Change maternal Hb conc. values
            mu_a_gen.m_c = np.array([x * 1.05 for x in mu_a_gen.m_c])  # 5% above the original value

            # Create fetal conc. 5% above, below and at the given value
            for i in range(3):
                if i == 0:
                    pass
                elif i == 1:  # 5% above
                    mu_a_gen.f_c = np.array([x * 1.05 for x in mu_a_gen.f_c])
                else:  # 5% below
                    mu_a_gen.f_c = np.array([x * 0.95 for x in mu_a_gen.f_c])

                all_mu_a_mom, all_mu_a_fetus = mu_a_gen.generate()
                base_mu_map = MU_MAP_BASE1.copy() if wave_int == 1 else MU_MAP_BASE2.copy()

                # Get intensity
                intensity_column = FastDataGen(Path(file), base_mu_map, 0, 3, all_mu_a_mom, all_mu_a_fetus).run()
                # PS: Don't forget to call run on the FastDataGen

                # Create the annotation columns
                tmp_df = TMPColumnGenerator(
                    mu_a_gen, len(intensity_column), SDD_list, wave_int, uterus_thickness, maternal_wall_thickness
                ).generate()

                # Putting it all together
                intensity_df = pd.DataFrame({"Intensity": intensity_column})
                intensity_df = pd.concat([tmp_df, intensity_df], axis=1)

                # Append new data at the end of old data (if any)
                old_df = pd.DataFrame()
                if os.path.exists(output_file):
                    old_df = pd.read_pickle(output_file)

                combined_df = pd.concat([old_df, intensity_df], axis=0, ignore_index=True)
                combined_df.to_pickle(output_file)

                # Cleanup dfs
                del old_df
                del combined_df
                del intensity_df

                # Update the progress bar
                pbar.update(1)
