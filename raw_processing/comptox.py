'''This module contains functions for loading/parsing raw data from the 
U.S. EPA CompTox Chemistry Dashboard. 

These data were batch downloaded and stored as either .CSV or .XLSX files 
from https://comptox.epa.gov/dashboard/
'''
import pandas as pd 
import numpy as np 
import warnings

from .utilities import inverse_log10_transform

#region: opera_test_predictions_from_csv
def opera_test_predictions_from_csv(
        predictions_path, index_col, chemicals_to_exclude=None, 
        columns_to_exclude=None, log10_pat=None, write_path=None):
    '''Load and parse the "predicted and intrinsic" chemical properties from 
    a CSV file.

    For columns in which there are predictions from both QSAR models, OPERA 
    and TEST, the OPERA predictions are returned.

    Parameters
    ----------
    predictions_path :  str
        File path to the raw data.    
    index_col : str
        Name of the chemical identifier in the headers to use as index.
    chemicals_to_exclude : list of str (optional)
        List of chemical identifiers to exclude from the return.
    columns_to_exclude :  list of str (optional)
        Names of columns to exclude from the return.
    log10_pat : str (optional)
        Substring in the columns indicating log10-transformed features, will 
        be used to inverse-transform these features.
    write_path : str (optional)
        Path to write the return as a CSV file.

    Returns
    -------
    pandas.DataFrame
    '''
    predictions = pd.read_csv(
        predictions_path, 
        index_col=index_col)

    if columns_to_exclude is not None:
        predictions = predictions.drop(
            columns_to_exclude, axis=1, errors='ignore')
    if chemicals_to_exclude is not None:
        predictions = predictions.drop(
            chemicals_to_exclude, errors='ignore')

    # Initialize a mapping {original key --> stripped key}.
    column_mapper_for = {}
    for model in ['OPERA', 'TEST']:
        model_columns = predictions.columns[
            predictions.columns.str.contains(model)]
        column_mapper_for[model] = {
            col: strip_model_name(col, model) for col in model_columns}

    columns_in_both_models = set.intersection(
        set(column_mapper_for['TEST'].values()),
        set(column_mapper_for['OPERA'].values()))

    test_columns_in_opera = [
        col for col, col_stripped in column_mapper_for['TEST'].items() 
        if col_stripped in columns_in_both_models]

    predictions = predictions.drop(test_columns_in_opera, axis=1)

    if log10_pat is not None:
        # Transform features into their original scales. Assume base-10.
        predictions = inverse_log10_transform(predictions, log10_pat)

    if write_path is not None:
        predictions.to_csv(write_path)

    return predictions
#endregion

#region: strip_model_name
def strip_model_name(string, model_name):
    '''Helper function to strip the model name from the column name.

    See Also
    --------
    opera_test_predictions_from_csv
    '''
    string_stripped = (
        string.removeprefix(model_name + '_')
        .removesuffix('_' + model_name + '_PRED'))
    return string_stripped
#endregion

# TODO: Could make index_col argument more flexible and optional.
#region: chemical_properties_from_excel
def chemical_properties_from_excel(
        dis_props_file, index_col, chemicals_to_exclude=None, 
        min_for_column=None, log10_pat=None, write_path=None):
    '''Parse the raw export of physical-chemical properties.

    Aggregate the values across data sources, giving preference to 
    experimental/measured data over predicted data.

    Parameters
    ----------
    dis_props_file : str
        File path to the raw data.
    index_col : str
        Name of the chemical identifier in the headers to use as index.
    chemicals_to_exclude : list of str (optional)
        List of chemical identifiers to exclude from the return.
    min_for_column : dict (optional)
        Mapping {column name --> minimum value}, can be used to set/ensure
        plausible ranges.
    log10_pat : str (optional)
        Substring in the columns indicating log10-transformed features, will 
        be used to inverse-transform these features.
    write_path : str (optional)
        Path to write the return as a CSV file.

    Returns
    -------
    pandas.DataFrame
    '''
    dis_props = pd.read_excel(
        dis_props_file, sheet_name='Chemical Properties')

    if chemicals_to_exclude is not None:
        dis_props = dis_props.set_index(index_col, drop=False)
        dis_props = dis_props.drop(chemicals_to_exclude, errors='ignore')

    dis_props['VALUE'] = (
        dis_props['VALUE']
        .replace(' ', np.nan)
        .astype('float'))
    
    if min_for_column is not None:
        dis_props = ensure_plausible_ranges(dis_props, min_for_column)
    
    # Prepare the index for group-by operations.
    index_cols = [index_col, 'TYPE', 'NAME']
    dis_props = dis_props.set_index(index_cols)

    # Aggregate the values across data sources.
    agg_props = (
        dis_props
        .groupby(index_cols)
        ['VALUE']
        .quantile(0.5)
        .unstack(level=['TYPE', 'NAME']))

    ## Use experimental values if available. Else, predicted. 

    agg_props_exp_imputed = (
        agg_props['experimental']
        .fillna(agg_props['predicted']))
    
    diff_pred_columns = (
        agg_props['predicted'].columns
        .difference(agg_props['experimental'].columns))
    
    agg_props_pred_remaining = (
        agg_props['predicted'][diff_pred_columns])

    agg_props = pd.concat(
        [agg_props_exp_imputed, agg_props_pred_remaining], 
        axis=1)

    agg_props.columns.name = None

    if log10_pat is not None:
        # Transform features into their original scales. Assume base-10.
        agg_props = inverse_log10_transform(agg_props, log10_pat)

    if write_path is not None:
        agg_props.to_csv(write_path)

    return agg_props
#endregion

#region: ensure_plausible_ranges
def ensure_plausible_ranges(dis_props, min_for_column):
    '''Helper function set/ensure plausible ranges.

    See Also
    --------
    chemical_properties_from_excel

    Parameters
    ----------
    min_for_column : dict (optional)
        Mapping {column name --> minimum value}, can be used to set/ensure
        plausible ranges.
    '''
    for column, theoretical_min in min_for_column.items():
        
        where_fails_criteria = (
            (dis_props['NAME'] == column) 
            & (dis_props['VALUE'] < theoretical_min))

        # Replace the value with "missing."
        dis_props.loc[where_fails_criteria, 'VALUE'] = np.nan
        
    return dis_props
#endregion

#region: chemicals_to_exclude_from_qsar
def chemicals_to_exclude_from_qsar(qsar_ready_smiles):
    '''Return a list of chemicals (str) that are inappropriate for QSAR
    modeling. 

    Identify chemicals that are missing "QSAR-ready" SMILES.

    Parameters
    ----------
    qsar_ready_smiles : pandas.Series
        "QSAR-Ready" SMILES for each chemical.
        Index = chemical identifiers, values = SMILES strings.

    References
    ----------
    - https://comptox.epa.gov/dashboard/
    - https://github.com/kmansouri/QSAR-ready

    Notes
    -----
    This standardization workflow by Mansouri was used by the OPERA model and 
    includes various filters for inorganics, mixtures, etc. As such, we assume
    that chemicals which lack a QSAR-ready SMILES did not pass these filters 
    and are therefore inappropriate for QSAR modeling.
    '''
    # Check whether the input data is appropriate. 
    name = qsar_ready_smiles.name
    # Convert string to uppercase to cover more cases.
    if name.upper() != 'QSAR_READY_SMILES':
        warnings.warn(f'Name, "{name}," does not suggest QSAR-Ready Smiles')

    where_missing = qsar_ready_smiles.isna()
    return list(qsar_ready_smiles.loc[where_missing].index)
    #endregion