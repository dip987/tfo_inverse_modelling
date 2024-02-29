"""
Functions to load data processed through the data pipeline for downstream processing.
"""
from pathlib import Path
from typing import Tuple, List
import pandas as pd

def load_pipeline_data(name: str) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Load data processed through the data pipeline.
    :param name: str: Name of the data pipeline to load. This should just be a the base name, no extensions, no 
    filepaths. The filepath is assumed to be the default location for the data pipeline. 'data/processed_data/name'
    :return: Tuple[List[str], List[str], pd.DataFrame]: Tuple containing the list of input features, labels and the 
    data as a DataFrame
    """
    data_base_path = Path(__file__).parent.parent.parent.resolve() / 'data' / 'processed_data'
    config_file = data_base_path / f'{name}.json'
    data_path = data_base_path / f'{name}.pkl'
    
    # Setup the variable types
    features: List[str]
    labels: List[str]
    data: pd.DataFrame
    
    # Put a type guard around the read_json and read_pickle functions
    features = pd.read_json(config_file)['features'].tolist()
    labels = pd.read_json(config_file)['labels'].to_list()
    data = pd.read_pickle(data_path)
    
    return features, labels, data