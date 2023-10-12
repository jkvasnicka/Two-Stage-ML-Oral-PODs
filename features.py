'''Functions for feature processing.
'''

import pandas as pd
from scipy.stats import skewtest
import warnings

# TODO: Split into submodules? 

#region: center_scale
def center_scale(X, u, s):
    '''Standardize a dataset (X).

    Center to u and component-wise scale with s.
    '''
    return (X - u) / s
#endregion

#region: median_absolute_deviation
def median_absolute_deviation(x):
    '''Return the median absolute deviation for a pandas.Series or DataFrame
    object (axis=0).

    Could be used to scale median-centered data. More robust to outliers than 
    standard deviation.
    '''
    q = 0.5
    dev = x - x.quantile(q)
    return dev.abs().quantile(q)
#endregion

#region: correlated_columns
def correlated_columns(
        data, thres=0.9, method=None, non_continuous_cols=None, errors=None):
    '''Return a list of columns with correlation coefficients that exceed the 
    threshold.

    Applies only to columns with continuous data (float). 

    Parameters
    ----------
    data : pandas.DataFrame
    thres : float
        Columns with correlation coefficients that exceed this value will be
        dropped (one column per pair).
    method : str (optional)
        Method of correlation for pandas.DataFrame.corr. Default 'pearson'.
    non_continuous_cols : single label or list-like (optional)
        Non-continuous column labels to drop. 
    errors : {'ignore', 'raise'}, default 'raise'
        If 'ignore', suppress error and only existing labels are dropped.
    '''
    if method is None:
        method = 'pearson'

    data = subset_continuous(data, non_continuous_cols, errors)

    # Initialize the full correlation matrix.
    corr_matrix = data.corr(method=method).abs()
    # Ignore the main diagonal.
    N = len(corr_matrix)
    corr_matrix.to_numpy()[range(N), range(N)] = 0. 

    # Convert elements to boolean (True/False).
    bool_matrix = corr_matrix > thres
    columns_most_to_least_correlated = (
        bool_matrix
        .sum(axis=1)
        .sort_values(ascending=False)
        .index)
    bool_matrix = bool_matrix.loc[columns_most_to_least_correlated]

    ## Initialize lists of correlated columns.
    i = 0
    top_row = bool_matrix.iloc[i]
    # Boolean indexing returns only True elements.
    all_correlated_columns = list(top_row.loc[top_row].index)
    correlated_columns = all_correlated_columns.copy()
    while correlated_columns:
        bool_matrix = bool_matrix.drop(correlated_columns)
        bool_matrix = bool_matrix.drop(correlated_columns, axis=1) 
        ## Repeat for the feature with the next most correlations.
        i += 1
        top_row = bool_matrix.iloc[i]
        correlated_columns = list(top_row.loc[top_row].index)
        all_correlated_columns += correlated_columns

    return all_correlated_columns
#endregion

#region: columns_missing_exceeding
def columns_missing_exceeding(data, thres=0.1):
    '''Return a list of columns with missing values (NaN) exceeding the 
    threshold proportion (0., 1.). 
    '''
    proportions_missing = data.isna().mean() 
    return list(data.loc[:, proportions_missing > thres].columns)
#endregion

#region: skewed_columns
def skewed_columns(data, alpha=0.05, **kwargs):
    '''Return a list of columns/rows (str) with significantly skewed values.

    Whether columns or rows depends on the specified axis.

    Parameters
    ----------
    alpha : int (optional)
        Statistical significance level for scipy.stats.skewtest. Features with 
        p-values less than this value are deemed "significant."
    kwargs 
        Key-word arguments for scipy.stats.skewtest().
    '''
    p_values = pd.Series(
        skewtest(
            data,
            **kwargs)
        .pvalue, 
        index=data.columns)
    return list(p_values.loc[p_values < alpha].index)
#endregion

#region: select_skewed
def select_skewed(data, alpha=0.05, **kwargs):
    '''Return a slice of the DataFrame with significantly skewed columns/rows.
    '''
    return data.loc[:, skewed_columns(data, alpha=alpha, **kwargs)]
#endregion

#region: constant_columns
def constant_columns(data):
    '''Return a list of columns with null variance or all NaN.

    Parameters
    ----------
    data : pandas.DataFrame
    '''
    where_constant = (data.var() == 0.) | (data.isna().all())
    return list(data.loc[:, where_constant].columns)
#endregion

#region: subset_continuous
def subset_continuous(data, non_continuous_cols=None, errors=None):
    '''Helper function to ensure data are continuous.

    Drop any non-continuous columns and issue a Userwarning if the resulting
    columns contain dtypes other than float.

    Parameters
    ----------
    data : pandas.DataFrame
    non_continuous_cols : single label or list-like (optional)
        Non-continuous column labels to drop. Any non-existent labels
    errors : {'ignore', 'raise'}, default 'raise'
        If 'ignore', suppress error and only existing labels are dropped.
    
    Returns
    -------
    DataFrame without the removed column labels.
    '''
    if errors is None:
        errors = 'raise'

    if non_continuous_cols is not None:
        data = data.drop(non_continuous_cols, axis=1, errors=errors)
    are_continuous(data)

    return data
#endregion

#region: are_continuous
def are_continuous(data):
    '''Issue a UserWarning if the data contains dtypes other than float.

    Parameters
    ----------
    data : pandas.DataFrame
    '''
    dtypes_not_float = [dt for dt in data.dtypes.unique() if dt != 'float']
    if dtypes_not_float:
        warnings.warn(
            f'data may contain non-continuous columns: dtypes {dtypes_not_float}')
#endregion