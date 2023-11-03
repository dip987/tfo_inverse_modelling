"""
Compare the results from the Fast datagen with the old process. This ensures there are not bugs on the new code with 
the results matching and also to check the execution time improvement
"""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from inverse_modelling_tfo.tools.intensity_gen_fast import FastDataGen, _find_intensity_column
from inverse_modelling_tfo.tools.intensity_datagen import intensity_from_raw, intensity_column_from_raw



data = Path(r"/home/rraiyan/simulations/tfo_sim/data/raw_dan_iccps_equispace_detector/fa_1_wv_1_sa_0.1_ns_1_ms_10_ut_5.pkl")

class TestValues(unittest.TestCase):
    # def test_values(self):
    #     mu_map = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm
    #     mu_0_values = np.array([0.1])
    #     mu_3_values = np.array([0.1, 0.3])
    #     intensity_fast = FastDataGen(data, np.array(list(mu_map.values())), 0, 3, mu_0_values, mu_3_values).run()
    #     intensity_old = []
    #     for mu_0 in mu_0_values:
    #         mu_map[1] = mu_0
    #         for mu_3 in mu_3_values:
    #             mu_map[3] = mu_3
    #             intensity_old.append(intensity_from_raw(data, mu_map)['Intensity'].to_numpy())
    #     intensity_old = np.array(intensity_old)
    #     intensity_old = intensity_old.reshape(-1)
        
    #     # Debug - Use this dataframe to check differences between valies
    #     df = pd.DataFrame({'A': intensity_fast, "C":intensity_old, 'Difference': np.abs(intensity_fast-intensity_old)})
        
    #     self.assertEqual(np.all(intensity_fast == intensity_old), True)
    
    def test_intensity_column(self):
        mu_map = {1: 0.0091, 2: 0.0158, 3: 0.0125, 4: 0.013}  # 735nm
        mu_0_values = 0.1
        mu_3_values = 0.3
        fast_gen = FastDataGen(data, np.array(list(mu_map.values())), 0, 3, mu_0_values, mu_3_values)
        intensity_fast = _find_intensity_column(fast_gen.fixed_pathlength, fast_gen.var1_pathlengths, 
                                                fast_gen.var2_pathlengths, fast_gen.fixed_mu_a, mu_0_values,
                                                mu_3_values)
        mu_map[1] = mu_0_values    
        mu_map[4] = mu_3_values
        intensity_old = intensity_column_from_raw(data, mu_map)['Intensity'].to_numpy().reshape(-1, 1)
        
        # Debug - Use this dataframe to check differences between valies
        df = pd.DataFrame({'Fast': intensity_fast.flatten(), "Old":intensity_old.flatten(), 
                           'Difference': np.abs(intensity_fast-intensity_old).flatten()})
        sim_data = pd.read_pickle(data)
        self.assertEqual(np.all(intensity_fast == intensity_old), True)

if __name__ == "__main__":
    unittest.main()
    