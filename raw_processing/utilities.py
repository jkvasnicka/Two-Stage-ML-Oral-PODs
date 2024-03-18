'''Module for shared utility functions for preprocessing raw data.
'''

import os 

#region: inverse_log10_transform
def inverse_log10_transform(data, log10_pat):
    '''Inverse transform any log10-transformed columns.
    
    Return a copy of the data in their original scales.

    Parameters
    ----------
    data : pandas.DataFrame
    log10_pat : str
        Pattern/substring indicating which columns to inverse transform.
    '''
    data = data.copy()

    log10_columns = get_columns_with_pattern(data.columns, log10_pat)
    data[log10_columns] = 10**data[log10_columns]
    return data
#endregion

#region: remove_pattern
def remove_pattern(columns, pattern):
    '''
    Helper function to remove a pattern (e.g., 'Log') from the column names.
    '''
    return columns.str.replace(pattern, '')
#endregion

#region: get_columns_with_pattern
def get_columns_with_pattern(columns, pattern):
    '''
    Helper function to get all columns that contain the specified pattern.
    '''
    pattern_columns =  [col for col in columns if pattern in col]
    if not pattern_columns:
        raise ValueError(
            f'Pattern, "{pattern}," not detected in any columns')
    else:
        return pattern_columns
#endregion

#region: tag_discrete_columns
def tag_discrete_columns(
        data, 
        discrete_columns, 
        suffix,
        validate_columns=True
        ):
    '''
    Tag discrete columns by adding a suffix.

    Parameters
    ----------
    data : pandas.DataFrame
    discrete_columns : list
        Columns to tag with a suffix.
    suffix : str
        Suffix to add to the column names.
    validate_columns : bool, default True
        If True, check that all discrete_columns are in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with renamed columns.
    '''
    if validate_columns:
        # Ensure all discrete_columns are in the DataFrame
        missing_cols = set(discrete_columns) - set(data.columns)
        if missing_cols:
            raise(ValueError(f'Columns {missing_cols} are not in the DataFrame'))
    
    return add_suffix_to_columns(data, discrete_columns, suffix)
#endregion

#region: add_suffix_to_columns
def add_suffix_to_columns(data, columns, suffix):
    '''
    Helper function to add a suffix to the specified columns.
    '''
    mapper = {col: col + suffix for col in columns}
    return data.rename(columns=mapper)
#endregion

#region: remove_suffix_from_columns
def remove_suffix_from_columns(data, suffix):
    '''
    Helper function to remove a suffix from all columns that have it.
    '''
    # Create a mapper only for columns that end with the suffix
    mapper = {
        col: col[:-len(suffix)] for col in data
        if col.endswith(suffix)
    }
    return data.rename(columns=mapper)
#endregion

# FIXME: There may be multiple copies of this function throughout the package
#region: ensure_directory_exists
def ensure_directory_exists(file_path):
    '''Check if the directory at `path` exists and if not, create it.'''
    # Extract the directory path from the file_path
    directory_path = os.path.dirname(file_path)
    
    # Check and create directories as needed
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path)
#endregion