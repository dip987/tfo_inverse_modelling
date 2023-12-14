"""
Function to generate optical propperties based on given parameters
"""


def get_tissue_mu_a(
    blood_volume_fraction: float,
    hb_concentration: float,
    saturation: float,
    wave_int: int,
) -> float:
    """
    Generate mu_a of a tissue layer based on given parameters (in mm^-1)

    Args:
        blood_volume_fraction (float): Fraction of blood in the tissue layer
        hb_concentration (float): Hb Concentration Levels
        saturation (float): Blood oxygen saturation
        wave_int (int): Wavelength integer (1 for 735nm, 2 for 850nm)

    Returns:
        float: mu_a of the tissue layer (in mm^-1)
    """
    wavelength_nm = 735 if wave_int == 1 else 850
    # Base mu_a that exists even without any pigments
    base_tissue_mu_a = 7.84e7 * wavelength_nm**-3.255  # in cm-1

    # Blood mu_a
    wave_index = wave_int - 1
    # Constants
    # 735nm, 850nm, 810nm
    # Values taken from Takatani(1987), https://omlc.org/spectra/hemoglobin/takatani.html
    # All values in cm-1/M
    epsilon_hhb = [412.0, 1058.0, 880.0]
    epsilon_hbo2 = [1464.0, 820.0, 888.0]

    # Convert concentration from g/dL to Moles/liter
    # per dL -> per L : times 10; g -> M : divide by grams per Mole
    # Assume HB and HBO2 have similar molar mass
    hb_concentration = hb_concentration * 10 / 64500  # in M/L
    # Notes: molar conc. is usually around 150/64500 M/L for regular human blood

    arterial_volume_fraction = blood_volume_fraction / 2
    venous_volume_fraction = blood_volume_fraction / 2
    venous_saturation = saturation * 0.75  # Venous saturation is usually 75% of arterial saturation

    # Use mu_a formula : mu_a = 2.303 * E * Molar Concentration
    mu_a_artery = (
        2.303 * hb_concentration * (saturation * epsilon_hhb[wave_index] + (1 - saturation) * epsilon_hbo2[wave_index])
    )  # in cm-1
    mu_a_venous = (
        2.303
        * hb_concentration
        * (venous_saturation * epsilon_hhb[wave_index] + (1 - venous_saturation) * epsilon_hbo2[wave_index])
    )  # in cm-1
    mu_a = venous_volume_fraction * mu_a_venous + arterial_volume_fraction * mu_a_artery + base_tissue_mu_a  # in cm-1
    mu_a = mu_a / 10  # Conversion to mm-1
    return mu_a

if __name__ == "__main__":
    print(get_tissue_mu_a(0.3, 12, 1.0, 1))
