"""Provides common functions that generate columns and handle dataframes"""
from pandas import DataFrame
from numpy import int32


def generate_sdd_column_(simulation_data: DataFrame) -> None:
    """Create an SDD(in mm) column in a simulation dataframe(From a single
    simulation). This assumes 2 columns 'X' and 'Y' in the data given in mm
    """
    varying_coordinate = 'X' if len(simulation_data['X'].unique()) > 1 else 'Y'
    fixed_coordinate = 'X' if varying_coordinate == 'Y' else 'Y'
    source_coordinate = simulation_data[fixed_coordinate][0]
    simulation_data['SDD'] = (
        simulation_data[varying_coordinate] - source_coordinate).astype(int32)  # in mm
