'''Helper functions for data input/output.
'''

import pandas as pd
from os import path 
from json import loads 

#region: combine_csv_files
def combine_csv_files(path_components_mapper, join=None, **csv_kwargs):
    '''Combine CSV files into a single DataFrame.

    Read each CSV file based on the specified path components. A dictionary is 
    used to map these components to keys for differentiating between data, 
    e.g., 'features' vs. 'targets'. 

    Parameters
    ----------
    path_components_mapper : dict[key] --> list of tuples
        The tuples represent path components for a given CSV file which are 
        joined using os.path.join(). 
    join {'inner', 'outer'}, default 'inner' 
        How to handle indexes on axis=0 (rows). The default returns the
        intersection.
    csv_kwargs : (optional)
        Key-word arguments for pandas.read_csv().

    Returns
    -------
    pandas.DataFrame with MultiIndex columns. 
        Level 0 of columns corresponds to the keys in path_components_mapper.
    '''
    if join is None:
        join = 'inner'
        
    # Initialize a container mapping key --> DataFrame.
    data_for = {}
    for k, path_components in path_components_mapper.items():
        # The first join combines DataFrames that share the key.
        data_for[k] = pd.concat(
            [pd.read_csv(path.join(*components), **csv_kwargs) 
            for components in path_components], 
            axis=1, 
            join=join
        )
    # The second join creates the MultiIndex columns.
    return pd.concat(data_for, axis=1, join=join)
#endregion

#region: json_to_dict
def json_to_dict(json_path):
    '''Load a JSON file as a dictionary.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    '''
    with open(json_path) as f:
        return loads(f.read())
#endregion