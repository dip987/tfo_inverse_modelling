from pandas import DataFrame
from numpy import log10
"""
Training data
fetal_mu_a = np.arange(0.05, 0.10, 0.005)
maternal_mu_a = np.arange(0.005, 0.010, 0.0005)
"""



def normalize_zero_one(data: DataFrame):
    """
    Normalize everything by the max to be between 0 and 1.
    """
    data['SDD'] = data['SDD'] / 20
    # data['Intensity'] = data['Intensity'] / 1e8  # Number of Photons
    data['Intensity'] = log10(data['Intensity']) / 8  # Number of Photons
    data['Uterus Thickness'] = data['Uterus Thickness'] / 8.
    data['Maternal Wall Thickness'] = data['Maternal Wall Thickness'] / 40.
    data['Wave Int'] = data['Wave Int'] - 1.0
    data['Fetal Mu_a'] = data['Fetal Mu_a']/0.1
    data['Maternal Mu_a'] = data['Maternal Mu_a']/0.01
    return data


