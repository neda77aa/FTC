# test file for pandas

from os.path import join
import pandas as pd
from typing import List, Callable, Union, Optional, Dict, Iterable

label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'binary': {'normal': 0.0, 'mild': 1.0, 'moderate': 1.0, 'severe': 1.0},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1}
}
class_labels: Dict[str, List[str]] = {
    'binary': ['Normal', 'AS'],
    'all': ['Normal', 'Mild', 'Moderate', 'Severe'],
    'not_severe': ['Not Severe', 'Severe'],
    'as_only': ['mild', 'moderate', 'severe'],
    'mild_moderate': ['mild', 'moderate'],
    'moderate_severe': ['moderate', 'severe']
}

def df_drop_by_value(df: pd.DataFrame(), column: str, value):
    """
    Remove entries of column X with the value v

    Parameters
    ----------
    df : pandas dataframe
    column : string, must be a valid column label
    value : value of the rows to remove

    Returns
    -------
    df2 : processed pandas dataframe

    """
    df2 = df[df[column] != value]
    return df2

def df_keep_by_value(df: pd.DataFrame(), column: str, value):
    """
    Keep entries of column X with the value v

    Parameters
    ----------
    df : pandas dataframe
    column : string, must be a valid column label
    value : value of the rows to keep

    Returns
    -------
    df2 : processed pandas dataframe

    """
    df2 = df[df[column] == value]
    return df2

dataset_root = r"D:\Datasets\as_tom"
dataset = pd.read_csv(join(dataset_root, 'annotations-all.csv'))

# append the dataset root onto every path
dataset['path'] = dataset['path'].map(lambda x: join(dataset_root, x))

# remove/modify columns in 'as_label' based on label scheme
scheme_name = 'as_only'
scheme = label_schemes[scheme_name]
dataset = dataset[dataset['as_label'].isin( scheme.keys() )]
dataset['as_label'] = dataset['as_label'].map(lambda x: scheme[x])

# count the number of datapoints per unique instance in as_label
v_counts = dataset['as_label'].value_counts()

