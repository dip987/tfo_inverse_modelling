"""
A set of routines that merge different physics-based data soruces for the physics based training.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd


def merge_l4_dist(l4_dist_path: Path, to_merge: pd.DataFrame, on_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Merge the L4 distribution data to the to_merge DataFrame on the on_column.

    The L4 distribution data changes with the body geometry. But remains unchanged with [Hb] and SaO2. This function
    merges the data based on these geometry parameters based on these on_columns parameters.
    :param l4_dist_path: Path to the L4 distribution data. Preferably a pkl. The directory should also contain the
    corresponding config. file. The config file should have the same name as the L4 distribution data file.
    :param to_merge: DataFrame to merge the L4 distribution data to.
    :param on_columns: Columns to merge the L4 distribution data on.
    """
    # Default on_columns
    if on_columns is None:
        on_columns = ["Maternal Wall Thickness"]

    # Sanity Checks
    assert l4_dist_path.exists(), f"Path {l4_dist_path} does not exist."
    assert l4_dist_path.suffix == ".pkl", f"Path {l4_dist_path} is not a pkl file."
    assert all(column in to_merge.columns for column in on_columns), f"Columns {on_columns} not present in Merging DF."

    l4_dist_term = pd.read_pickle(l4_dist_path)
    # Check if all the columns in on_columns are present in both the DataFrames
    assert all(column in l4_dist_term.columns for column in on_columns), f"Columns {on_columns} not present in L4 DF"

    # The distribution columns are titled as their bin center
    l4_dist_columns = filter(is_float, l4_dist_term.columns)
    # Sort by wavelength
    l4_dist_columns = [str(x) for x in l4_dist_columns]  # In case they are
    wv1_l4_dist = (l4_dist_term[l4_dist_term["Wave Int"] == 1])[["Maternal Wall Thickness"] + l4_dist_columns]
    wv2_l4_dist = (l4_dist_term[l4_dist_term["Wave Int"] == 2])[["Maternal Wall Thickness"] + l4_dist_columns]
    # Set indices to Maternal Wall Thickness for easy mapping
    wv1_l4_dist.set_index("Maternal Wall Thickness", inplace=True)
    wv2_l4_dist.set_index("Maternal Wall Thickness", inplace=True)

    # Start putting everytghing onto data
    data = to_merge.copy()
    for column in l4_dist_columns:
        data[column + " WV1"] = data["Maternal Wall Thickness"].map(wv1_l4_dist[column])
        data[column + " WV2"] = data["Maternal Wall Thickness"].map(wv2_l4_dist[column])

    return data


def is_float(element: str) -> bool:
    """
    Check if the given string can be converted to a float.
    """
    try:
        float(element)
        return True
    except ValueError:
        return False
