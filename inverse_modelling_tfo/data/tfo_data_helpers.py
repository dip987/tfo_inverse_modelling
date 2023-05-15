"""
Helper functions to work with TFO Probe data
"""
from typing import List
from pandas import DataFrame, concat, melt


def transform_tfo_data(data: DataFrame, sdd: List) -> DataFrame:
    """Transofrm PPG TFO Probe data into a long format which can then be used with the intensity 
    interpolation functions. The returned DataFrame contains 3 Columns: SDD, Wave Int and Intensity.
    (Note: The first n columns are treated as Wave Int 1.0. The last n columns as Wave Int 2.0; 
    where n is [len(sdd)])

    This function is compatible with both 4 and 5-detector setup. 
    Args:
        data (DataFrame): TFO PPG data loaded directly from the TFO_dataset package. Should have 
        column names like: ch1VoltsWL1, ... 
        sdd (List): A list containing the source to detector distances. The SDD column of the 
        return dataframe will be populated with this data


    Returns:
        DataFrame: DataFrame containing the PPG data in a long format with SDD and Wave Int columns
    """
    detector_count = len(sdd)
    wv1_data = data.iloc[:, :detector_count]
    wv2_data = data.iloc[:, -detector_count:]
    # Replace Column names with sdd values
    column1_mapper = dict(zip(wv1_data.columns, sdd))
    column2_mapper = dict(zip(wv2_data.columns, sdd))
    wv1_data.columns = [column1_mapper[col_name]
                        for col_name in wv1_data.columns]
    wv2_data.columns = [column2_mapper[col_name]
                        for col_name in wv2_data.columns]
    # Pivot (Using stack to keep the order I want, the same observation kept together)
    wv1_data =  wv1_data.stack().reset_index().rename(columns={'level_1':'SDD', 0:'Intensity'})
    wv1_data.drop('level_0', axis=1, inplace=True)
    wv1_data['Wave Int'] = 1.0

    wv2_data =  wv2_data.stack().reset_index().rename(columns={'level_1':'SDD', 0:'Intensity'})
    wv2_data.drop('level_0', axis=1, inplace=True)
    wv2_data['Wave Int'] = 2.0

    # combine
    # return wv1_data.append(wv2_data, ignore_index=True)
    return concat([wv1_data, wv2_data], ignore_index=True, axis=0)
