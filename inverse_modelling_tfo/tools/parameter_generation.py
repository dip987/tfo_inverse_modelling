"""
Methodically generate simulation parameter, which can later be passed onto any of the data generation code
"""


from itertools import product
from typing import Tuple, List, Union
import numpy as np
from pandas import DataFrame
from inverse_modelling_tfo.tools.optical_properties import get_tissue_mu_a


class MuAGenerator:
    """
    A convinience class to generate a list of all possible mu_a (maternal and fetal) based on the given initilization
    parameters. Call generate( ) to get maternal, fetal mu_a as a list of numpy array. This can later be passed on to
    simulation data generators. Note: the start & end points for the range tuples are inclusive.

    To alter this class's behaviour, extend it and update this class's internal variables (this includes m_s, m_c, f_s,
    f_c) while keeping the generate function unaltered.
    """

    def __init__(
        self,
        m_s_range: Tuple[float, float],
        m_s_count: int,
        m_c_range: Tuple[float, float],
        m_c_count: int,
        f_s_range: Tuple[float, float],
        f_s_count: int,
        f_c_range: Tuple[float, float],
        f_c_count: int,
        fetal_blood_volume_fraction: float,
        maternal_blood_volume_fraction: float,
        wave_int: int,
    ) -> None:
        super().__init__()
        self.m_s = np.linspace(m_s_range[0], m_s_range[1], num=m_s_count, endpoint=True)
        self.m_c = np.linspace(m_c_range[0], m_c_range[1], num=m_c_count, endpoint=True)
        self.f_s = np.linspace(f_s_range[0], f_s_range[1], num=f_s_count, endpoint=True)
        self.f_c = np.linspace(f_c_range[0], f_c_range[1], num=f_c_count, endpoint=True)
        self.maternal_blood_volume_fraction = maternal_blood_volume_fraction
        self.fetal_blood_volume_fraction = fetal_blood_volume_fraction
        self.wave_int = wave_int

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a list of all possible maternal and fetal mu_a based on the initialization saturation & Hb
        concentration and the given point counts.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a list of all maternal mu_a, fetal mu_a
        """
        all_maternal_combos = list(product(self.m_s, self.m_c))
        all_fetal_combos = list(product(self.f_s, self.f_c))
        all_maternal_mu_a = np.array(
            [
                get_tissue_mu_a(self.maternal_blood_volume_fraction, conc, sat, self.wave_int)
                for sat, conc in all_maternal_combos
            ]
        )
        all_fetal_mu_a = np.array(
            [
                get_tissue_mu_a(self.fetal_blood_volume_fraction, conc, sat, self.wave_int)
                for sat, conc in all_fetal_combos
            ]
        )
        return all_maternal_mu_a, all_fetal_mu_a


class ProximityMuAGenerator(MuAGenerator):
    """
    An extension of MuAGenerator where you can include additional concentration values. The additional concentration
    values are some fraction larger than each of the values generated from MuAGenerator.

    Note: The additional concentration filed can be an empty list. In that case, this class behaves exactly like a
    [MuAGenerator] class.

    Example
    -------
    MuGenerator Hb Conc. values : 1, 2, 3
    fraction : [0.05, -0.05]
    additional Hb Conc. values : 1.05, 2.10, 3.15, 0.95, 1.90, 2.85
    """

    def __init__(
        self,
        m_s_range: Tuple[float, float],
        m_s_count: int,
        m_c_range: Tuple[float, float],
        m_c_count: int,
        f_s_range: Tuple[float, float],
        f_s_count: int,
        f_c_range: Tuple[float, float],
        f_c_count: int,
        wave_int: int,
        maternal_blood_volume_fraction: float,
        fetal_blood_volume_fraction: float,
        maternal_proximity_fractions: List[float],
        fetal_proximity_fractions: List[float],
    ) -> None:
        super().__init__(
            m_s_range,
            m_s_count,
            m_c_range,
            m_c_count,
            f_s_range,
            f_s_count,
            f_c_range,
            f_c_count,
            fetal_blood_volume_fraction,
            maternal_blood_volume_fraction,
            wave_int,
        )
        all_maternal_concs = [self.m_c.copy()]
        for fraction in maternal_proximity_fractions:
            additional_concs = self.m_c * (1 + fraction)
            all_maternal_concs.append(additional_concs)
        self.m_c = np.array(all_maternal_concs).flatten()

        all_fetal_concs = [self.f_c.copy()]
        for fraction in fetal_proximity_fractions:
            additional_concs = self.f_c * (1 + fraction)
            all_fetal_concs.append(additional_concs)
        self.f_c = np.array(all_fetal_concs).flatten()


class TMPColumnGenerator:
    """
    A convinience class to generate the TMP columns corresponding intensity data generated using Mu_a from a
    MuAGenerator class.

    Call generate( ) to get the TMP columns as a DataFrame which can be concatenated horizontally with the intensity
    data.

    """

    def __init__(
        self,
        mu_a_gen: MuAGenerator,
        generated_data_length: int,
        sdd_list: np.ndarray,
        wave_int: int,
        uterus_thickness: Union[float, int],
        maternal_wall_thickness: Union[float, int],
    ) -> None:
        self.mu_a_gen = mu_a_gen
        self.num_rows = generated_data_length
        self.sdd_list = sdd_list
        self.wave_int = wave_int
        self.uterus_thickness = uterus_thickness
        self.maternal_wall_thickness = maternal_wall_thickness
        self.column_names = [
            "Wave Int",
            "SDD",
            "Uterus Thickness",
            "Maternal Wall Thickness",
            "Maternal Hb Concentration",
            "Maternal Saturation",
            "Fetal Hb Concentration",
            "Fetal Saturation",
        ]

    def generate(self) -> DataFrame:
        """
        Generate a DataFrame with the TMP annotation columns corresponding to the intensity data generated using Mu_a
        """
        all_mu_a_mom, _ = self.mu_a_gen.generate()
        # Get additional properties for annotating the dataframe
        all_sat_con_fetus = list(product(self.mu_a_gen.f_s, self.mu_a_gen.f_c))
        all_sat_con_mom = list(product(self.mu_a_gen.m_s, self.mu_a_gen.m_c))
        sdd_column = np.tile(self.sdd_list, self.num_rows // len(self.sdd_list))
        wave_int_column = self.wave_int * np.ones((self.num_rows,))
        uterus_thickness_column = self.uterus_thickness * np.ones((self.num_rows,))
        maternal_wall_thickness_column = self.maternal_wall_thickness * np.ones((self.num_rows,))
        fetal_saturation_column = np.tile(
            np.repeat(np.array([x[0] for x in all_sat_con_fetus]), len(self.sdd_list)), len(all_mu_a_mom)
        )
        fetal_concentration_column = np.tile(
            np.repeat(np.array([x[1] for x in all_sat_con_fetus]), len(self.sdd_list)), len(all_mu_a_mom)
        )
        mom_saturation_column = np.repeat(
            np.array([x[0] for x in all_sat_con_mom]), self.num_rows // len(all_sat_con_mom)
        )
        mom_concentration_column = np.repeat(
            np.array([x[1] for x in all_sat_con_mom]), self.num_rows // len(all_sat_con_mom)
        )

        return DataFrame(
            {
                "Wave Int": wave_int_column,
                "SDD": sdd_column,
                "Uterus Thickness": uterus_thickness_column,
                "Maternal Wall Thickness": maternal_wall_thickness_column,
                "Maternal Hb Concentration": mom_concentration_column,
                "Maternal Saturation": mom_saturation_column,
                "Fetal Hb Concentration": fetal_concentration_column,
                "Fetal Saturation": fetal_saturation_column,
            }
        )
