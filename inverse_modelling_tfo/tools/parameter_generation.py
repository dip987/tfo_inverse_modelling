"""
Methodically generate simulation parameter, which can later be passed onto any of the data generation code
"""


from itertools import product
from typing import Tuple, List
import numpy as np


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
    E_HB = [412.0, 1058.0, 880.0]
    E_HBO2 = [1464.0, 820.0, 888.0]

    # Convert concentration from g/dL to Moles/liter
    # per dL -> per L : times 10; g -> M : divide by grams per Mole
    # Assume HB and HBO2 have similar molar mass
    concentration = concentration * 10 / 64500  # in M/L
    # Notes: molar conc. is usually around 150/64500 M/L for regular human blood

    # Use mu_a formula : mu_a = 2.303 * E * Molar Concentration
    mu_a = 2.303 * concentration * (saturation * E_HB[wave_index] + (1 - saturation) * E_HBO2[wave_index])  # in cm-1

    mu_a = mu_a / 10  # Conversion to mm-1
    return mu_a


class MuAGenerator:
    """
    A convinience class to generate a list of all possible mu_a (maternal and fetal) based on the given initilization
    parameters. Call generate( ) to get maternal, fetal mu_a as a list of numpy array. This can later be passed on to
    simulation data generators. Note: the start & end points for the range tuples are inclusive.

    To alter this class's behaviour, extend it and update this class's internal variables while keeping the generate
    function unaltered.
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
    ) -> None:
        super().__init__()
        self.m_s = np.linspace(m_s_range[0], m_s_range[1], num=m_s_count, endpoint=True)
        self.m_c = np.linspace(m_c_range[0], m_c_range[1], num=m_c_count, endpoint=True)
        self.f_s = np.linspace(f_s_range[0], f_s_range[1], num=f_s_count, endpoint=True)
        self.f_c = np.linspace(f_c_range[0], f_c_range[1], num=f_c_count, endpoint=True)
        self.wave_int = wave_int

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a list of all possible maternal and fetal mu_a based on the initialization saturation & Hb
        concentration and the given point counts.

        Returns:
            Tuple[np.ndarray, np.ndarray]: a list of all maternal mu_a, fetal mu_a
        """
        all_maternal_combos = list(product(self.m_s, self.m_c))
        all_fetal_combos = list(product(self.f_s, self.f_c))
        all_maternal_mu_a = np.array([get_mu_a(sat, conc, self.wave_int) for sat, conc in all_maternal_combos])
        all_fetal_mu_a = np.array([get_mu_a(sat, conc, self.wave_int) for sat, conc in all_fetal_combos])
        return all_maternal_mu_a, all_fetal_mu_a


class ProximityMuAGenerator(MuAGenerator):
    """
    An extension of MuAGenerator where you can include additional concentration values. The additional concentration 
    values are some fraction larger than each of the values generated from MuAGenerator.
    
    Example
    -------
    MuGenerator values : 1, 2, 3
    fraction : [0.05, -0.05]
    additional values : 1.05, 2.10, 3.15, 0.95, 1.90, 2.85
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
        maternal_proximity_fractions: List[float],
        fetal_proximity_fractions: List[float],
    ) -> None:
        super().__init__(
            m_s_range, m_s_count, m_c_range, m_c_count, f_s_range, f_s_count, f_c_range, f_c_count, wave_int
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
