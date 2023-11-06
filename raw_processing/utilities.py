'''Module for shared utility functions for preprocessing raw data.
'''

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

    log_columns = get_columns_with_pattern(data.columns, log10_pat)
    if not log_columns:
        raise ValueError(
            f'log10 pattern, "{log10_pat}," not detected in any columns')
    
    data = apply_inverse_log_transform(data, log_columns)
    
    data.columns = data.columns.str.replace(log10_pat, '')
    return data
#endregion

#region: get_columns_with_pattern
def get_columns_with_pattern(columns, pattern):
    '''
    Helper function to get all columns that contain the specified pattern.
    '''
    return [col for col in columns if pattern in col]
#endregion

#region: apply_inverse_log_transform
def apply_inverse_log_transform(data, columns):
    '''
    Helper function to apply inverse log10 transformation to specified 
    columns.
    '''
    data[columns] = 10**data[columns]
    return data
#endregion