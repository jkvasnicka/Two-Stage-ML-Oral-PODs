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

    log_columns = [c for c in data if log10_pat in c]
    if not log_columns:
        raise ValueError(
            f'log10 pattern, "{log10_pat}," not detected in any columns')
    data.loc[:, log_columns] = 10**data[log_columns]
    data.columns = data.columns.str.replace(log10_pat, '')
    return data
#endregion
