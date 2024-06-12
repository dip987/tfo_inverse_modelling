import pandas as pd
import numpy as np


def custom_hist_edge_generator(bin_count: int, max_val: float):
    """
    Generate bin edges for our custom Histogram function.
    :param bin_count: Number of bins to generate.
    :param max_val: Maximum value to consider.
    """
    first_left_edge = 0.0
    first_right_edge = 1.0  # Resolution of 1mm -> No data captured should be less than 1mm
    last_right_edge = max_val
    middle_edges = np.logspace(np.log10(first_right_edge), np.log10(last_right_edge), num=bin_count, base=10.0)
    return np.concatenate(([first_left_edge], middle_edges))


def custom_histogram(data: pd.Series, bin_count: int = 10, max_val: float = 345.0):
    """
    Bins L4 ppath using a custom implementation of histogram.
    :param data: The L4 ppath data as pandas Series. The data is assumed to be in mm.
    :param bin_count: Number of bins to use. Default is 10.
    :param max_val: Maximum value to consider. Any pathlength beyond this is dropped. Default is 345.0 mm.
    :return: A tuple of two numpy arrays. (Histogram, Bin Centers)

    Custom Histogram:
    ----------------
    The custom histogram function minimizes the reconstruction error in exp(-L). This is the term that care about in
    the physics based loss for inverse-modeling. The custom histogram function is implemented as follows:

        1. The first bin is always between [0, 1) mm
        2. However, the first bin center is always considered at 0. This accomodates the very large distribution spike
        seen at 0. Moving the center to 0.5 introduces a very large error.
        3. The rest of the bins are logarithmically spaced between [1, max_val] mm.
        4. Any data point beyond max_val is dropped. The reason being exp(-L) for large values of L is very close to 0.
        We default to 345. This is because we ultimately care about exp(-\\mu*L) where \\mu is the absorption coefficient.
        The average value of \\mu is around 0.04 mm^-1. So, exp(-0.04*345) produces 10^-6. This is a good threshold.

    """
    bin_edges = custom_hist_edge_generator(bin_count, max_val=max_val)
    temp_data = data[data <= max_val]  # Drop outliers on the right
    hist, _ = np.histogram(temp_data, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist = hist / len(data) # Normalize the histogram
    bin_centers[0] = 0.0  # Force the first bin center to be 0. Massively reduces the error!
    return hist, bin_centers
