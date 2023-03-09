from typing import Tuple
import numpy as np
from pandas import DataFrame
from pandas import read_pickle

def interpolate_exp_chunk(data: DataFrame, weights : Tuple[float, float] = (1, -10)) -> np.array:
    """Exponentially interpolate to chunk of data(20 sets of SDD, preferably) to create a denoised version of the Intensity. The interpolation
    uses a weighted version of linear regression. (More info here : https://en.wikipedia.org/wiki/Weighted_least_squares)

    Args:
        data (DataFrame): A chunk of the data. The whole chunk will be used for interpolation
        weights (Tuple[float, float]): Weights used during interpolation. Supply the weight of the first and last element. Logarithmically
        picks the middle weights. Defaults to (1, -10)

    Returns:
        np.array: returns the interpolated data as a (n x 1) numpy array
    """
    X = np.ones((len(data), 3))  # One column for SDD and one for the bias
    X[:, 0] = data['SDD'].to_numpy()
    X[:, 1] = np.sqrt(data['SDD'].to_numpy())
    Y = np.log(data['Intensity'].to_numpy()).reshape(-1, 1)
    W = np.diag(np.logspace(weights[0], weights[1], num=len(Y)))   # Define the weight
    beta_hat = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ Y # Solve
    y_hat = X @ beta_hat
    return np.exp(y_hat)


def interpolate_exp(data: DataFrame, weights : Tuple[float, float] = (1.0, -10), sdd_chunk_size: int = 20) -> DataFrame:
    """Exponentially interpolate to chunk of data(20 sets of SDD, preferably) to create a denoised version of the Intensity. The interpolation
    uses a weighted version of linear regression. (More info here : https://en.wikipedia.org/wiki/Weighted_least_squares)

    Args:
        data (DataFrame): Simulation data loaded in an orderly fashion. (The code expects all 20 SDDs to be in a sequence)
        weights (Tuple[float, float], optional): Weights used during interpolation. Supply the weight of the first and last element. Logarithmically
        picks the middle weights. Defaults to (1.0, -10). (i.e, 10^1 to 10^-10). 
        sdd_chunk_size (int, optional): In case there are a different number of SDD values. Defaults to 20.

    Returns:
        DataFrame: Returns a dataframe with a new column 'Interpolated Intensity', holding the interpolations 
    """
    interpolated_intensity = None
    for i in range(len(data)//sdd_chunk_size):
        data_chunk = data.iloc[sdd_chunk_size*i : sdd_chunk_size * (i + 1), :]
        y_hat = interpolate_exp_chunk(data_chunk, weights)
        if interpolated_intensity is None:
            interpolated_intensity = y_hat
        else:
            interpolated_intensity = np.vstack([interpolated_intensity, y_hat])
    data['Interpolated Intensity'] = interpolated_intensity
    return data
    
    
if __name__ == "__main__":
    loaded_data = read_pickle(r'/home/rraiyan/personal_projects/tfo_inverse_modelling/data/intensity/test_data.pkl')
    filtered_data = interpolate_exp(loaded_data, weights=[1, -10])
    
    print("HALT")
        
        
        
        
    


    
