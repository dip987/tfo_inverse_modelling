"""
Functions to generate detector intensity from photon path distributions.
"""

from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def create_sdd_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe containing 'X' and 'Y' columns, calculate the Source to Detector Distance (SDD) and add it as a
    new column.
    :param df: DataFrame containing 'X' and 'Y' columns.
    :return: A copy of the dataframe with 'SDD' column but without the 'X'/'Y'/'Z' columns.
    """
    ## Sanity Check
    assert "X" in df.columns and "Y" in df.columns, "X and Y columns not found in the dataframe!"

    df_temp = df.copy()
    # The way my RAW files are setup, only one of the X or Y columns will have more than one unique value
    varying_coordinate = "X" if len(df["X"].unique()) > 1 else "Y"
    fixed_coordinate = "X" if varying_coordinate == "Y" else "Y"
    source_coordinate = df[fixed_coordinate][0]
    df_temp["SDD"] = (df[varying_coordinate] - source_coordinate).astype(np.int32)
    df_temp.drop(["X", "Y"], axis=1, inplace=True)
    return df_temp


def intensity_from_distribution(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Convert a CSV containing photon path densities into detector intensity for any given set of absorption co-efficients.
    Uses the method described in Fredriksson et al. 2012. For a detailed description of each step in the code please
    look at the notebook named "data_format_exploration.ipynb".

    The mu_map should contain {layer number(int) : Mu(float) in mm-1}.
    The output is a dataframe with two columns SDD and Intensity
    """

    simulation_data = pd.read_csv(
        file_path, dtype={"Ppath Medium": np.int32, "Deepest Layer": np.int32, "Count": np.int32}
    )
    # Convert X,Y, Z co-ordinates to SDD using a very hacky method
    simulation_data = create_sdd_column(simulation_data)

    # Normalization
    normalization_factor = simulation_data.groupby(["Deepest Layer", "SDD"])["Count"].transform("sum")
    normalization_factor_grouped = simulation_data.groupby(["Deepest Layer", "SDD"])["Count"].sum()
    simulation_data["Count"] = simulation_data["Count"] / normalization_factor

    # Weighted Path Length
    simulation_data["Weighted Ppath"] = simulation_data["Count"] * np.exp(
        -simulation_data["Bin Center"] * unitinmm * [mu_map[medium] for medium in simulation_data["Ppath Medium"]]
    )

    # Taking the area under the distribution
    simulation_data = simulation_data.groupby(["Ppath Medium", "Deepest Layer", "SDD"], sort=False, as_index=False)[
        "Weighted Ppath"
    ].sum()

    # Multiplying distribution from each layer
    simulation_data = simulation_data.groupby(["Deepest Layer", "SDD"], sort=False)["Weighted Ppath"].prod()

    # Initial Intensity I_0
    simulation_data = simulation_data * normalization_factor_grouped

    # Sum up
    simulation_data = simulation_data.groupby(["SDD"]).sum()

    # Rename and create df
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()

    return simulation_data


def create_intensity_column(df: pd.DataFrame, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    For each row in the given dataframe, calculate the weighted intensity using the given absorption co-efficients.
    :param df: DataFrame containing 'SDD' and 'L{layer_num} ppath' columns.
    :param mu_map: Dictionary containing the absorption co-efficients for each layer. Should have the format
                    {layer number(int) : Mu(float) in mm-1}. The layer numbers should correspond to the column names in
                    the raw simulation file. e.g. L1 ppath, L2 ppath etc. -> {1: mu1, 2: mu2, ...}
    :param unitinmm: Multiplier to convert ppath in raw simulation file to mm. Default is 1.0.
    :return: The same dataframe with an additional 'Intensity' column containing the calculated intensity per row.

    Note: This function does not normalize the intensity values!
    """
    ## Sanity Check
    for layer in mu_map.keys():
        assert f"L{layer} ppath" in df.columns, f"L{layer} ppath column not found in the dataframe!"

    df_temp = df.copy()
    ppath_columns = [f"L{layer} ppath" for layer in mu_map.keys()]
    ppaths = torch.tensor(df[ppath_columns].values).cuda()
    mu_a_tensor = torch.tensor(list(mu_map.values()), dtype=torch.float32).reshape(1, -1).cuda()
    photon_intensity = torch.exp(torch.sum(-ppaths * unitinmm * mu_a_tensor, dim=1)).cpu().numpy()
    df_temp["Intensity"] = photon_intensity
    return df_temp


def intensity_from_raw(file_path: Path, mu_map: Dict[int, float], unitinmm: float = 1.0) -> pd.DataFrame:
    """
    Calculate the Per Detector Intensity from a given raw simulation file.

    :param file_path: Path to the raw simulation file.
    :param mu_map: Dictionary containing the absorption co-efficients for each layer. Should have the format
                    {layer number(int) : Mu(float) in mm-1}. The layer numbers should correspond to the column names in
                    the raw simulation file. e.g. L1 ppath, L2 ppath etc. -> {1: mu1, 2: mu2, ...}
    :param unitinmm: Multiplier to convert ppath in raw simulation file to mm. Default is 1.0.
    :return: DataFrame containing the 'SDD' and 'Intensity' columns.

    Note: This function does not normalize the intensity values!
    """
    simulation_data = pd.read_pickle(file_path)

    # Convert X,Y, Z co-ordinates to SDD
    simulation_data = create_sdd_column(simulation_data)

    # Calculate Intensity per Row
    simulation_data = create_intensity_column(simulation_data, mu_map, unitinmm)

    # Sum per detector (Creates a pd.Series)
    simulation_data = simulation_data.groupby(["SDD"])["Intensity"].sum()

    # Convert this series to a dataframe
    simulation_data.name = "Intensity"
    simulation_data = simulation_data.to_frame().reset_index()

    return simulation_data
