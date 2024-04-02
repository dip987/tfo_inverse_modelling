"""
Functions to load data processed through the data pipeline for downstream processing.
"""

import json
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
    data_base_path = Path(__file__).parent.parent.parent.resolve() / "data" / "processed_data"
    config_file = data_base_path / f"{name}.json"
    data_path = data_base_path / f"{name}.pkl"

    # Sanity Checks
    if not config_file.exists():
        raise FileNotFoundError(f"Config file for {name} not found at {config_file}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file for {name} not found at {data_path}")

    # Setup the variable types
    features: List[str]
    labels: List[str]
    data: pd.DataFrame

    # Load configs
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        features = config["features"]
        labels = config["labels"]

    # Load data
    data = pd.read_pickle(data_path)

    return features, labels, data
