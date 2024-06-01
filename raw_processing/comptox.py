'''
This module contains functions for loading and processing raw data from the 
U.S. EPA CompTox Chemistry Dashboard. 

These data were batch downloaded and stored as either .CSV or .XLSX files 
from the following source: https://comptox.epa.gov/dashboard/
'''
import pandas as pd 

from . import utilities

#region: opera_test_predictions_from_csv
def opera_test_predictions_from_csv(
        predictions_path, 
        index_col, 
        columns_to_exclude=None, 
        log10_pat=None, 
        write_path=None
        ):
    '''Load and parse the "predicted and intrinsic" chemical properties from 
    a CSV file.

    For columns in which there are predictions from both QSAR models, OPERA 
    and TEST, the OPERA predictions are given priority.

    Parameters
    ----------
    predictions_path :  str
        File path to the raw data.    
    index_col : str
        Name of the chemical identifier in the headers to use as index.
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

    Notes
    -----
    CAUTION: This function does not filter out any chemicals (e.g., metals) and 
    therefore it is important to ensure data integrity before this function is
    used.
    '''
    predictions = pd.read_csv(
        predictions_path, 
        index_col=index_col
        )

    if columns_to_exclude is not None:
        predictions = predictions.drop(
            columns_to_exclude, axis=1, errors='ignore')

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
        predictions = utilities.inverse_log10_transform(
            predictions, 
            log10_pat
            )
        predictions.columns = utilities.remove_pattern(
            predictions.columns, 
            log10_pat
            )

    if write_path is not None:
        utilities.ensure_directory_exists(write_path)
        predictions.to_parquet(write_path)

    return predictions
#endregion

#region: strip_model_name
def strip_model_name(string, model_name):
    '''
    Helper function to strip the model name from the column name.

    See Also
    --------
    opera_test_predictions_from_csv
    '''
    string_stripped = (
        string.removeprefix(model_name + '_')
        .removesuffix('_' + model_name + '_PRED'))
    return string_stripped
#endregion