'''
This module contains functions for loading and processing raw data from the
OPERA 2.9 suite of models by Kamel Mansouri et al.

This module was designed to process batches of OPERA outputs where each batch
corresponds to a unique subdirectory of separate CSV files.

References
----------
https://github.com/NIEHS/OPERA/releases/tag/v2.9.1
'''

import pandas as pd 
import numpy as np
import os
import logging

from . import utilities

# NOTE: For backwards compatibility
def get_original_columns(columns_for_model):
    '''
    Return features columns in original order as manuscript submission. 

    This is only a temporary fix for backwards compatibility.
    '''
    return [item for sublist in columns_for_model.values() for item in sublist]

#region: process_all_batches
def process_all_batches(
        main_dir, 
        columns_for_model, 
        log_file_name, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        data_write_path=None, 
        flags_write_path=None
        ):
    '''
    Process all directories in the given main directory, where each directory 
    represents a batch of samples/chemicals.

    The results are concatenated across batches.

    Returns
    -------
    - Applicability domain flags (pandas.DataFrame)
    - Predictions (pandas.DataFrame)
    '''
    logging.basicConfig(
        filename=os.path.join(main_dir, log_file_name), 
        level=logging.INFO, 
        format='%(asctime)s %(message)s',
        filemode='w'  # overwrite any existing log file
        )

    # Initialize the containers for all batches.
    predictions, AD_flags = [], []

    ## Extract the data for each batch/subdirectory of chemicals
    for entry in os.listdir(main_dir):
        logging.info(f"Processing directory: {entry}")
        data_dir = os.path.join(main_dir, entry)

        if os.path.isdir(data_dir):
            try:
                batch_predictions, batch_AD_flags = (
                    extract_predictions_and_app_domains(
                        data_dir, 
                        columns_for_model, 
                        index_name=index_name, 
                        discrete_columns=discrete_columns, 
                        discrete_suffix=discrete_suffix, 
                        log10_pat=log10_pat
                    )
                )
                predictions.append(batch_predictions)
                AD_flags.append(batch_AD_flags)
            except Exception as e:
                logging.error(
                    f"Skipping directory {entry} due to error: {str(e)}")
                continue
    predictions = pd.concat(predictions)
    AD_flags = pd.concat(AD_flags)

    predictions, AD_flags = _perform_final_cleaning(predictions, AD_flags)

    _write_data(predictions, data_write_path)
    _write_data(AD_flags, flags_write_path)

    return predictions, AD_flags
#endregion

#region: _perform_final_cleaning
def _perform_final_cleaning(predictions, AD_flags):
    '''
    Perform final cleaning on the combined predictions.

    This includes any dropping rows with all missing values and removing any 
    duplicate index values.

    Parameters
    ----------
    predictions : pandas.DataFrame
        The DataFrame containing the combined predictions.
    AD_flags : pandas.DataFrame
        The DataFrame containing the corresponding applicability domain flags.

    Returns
    -------
    predictions : pandas.DataFrame
        The cleaned predictions DataFrame.
    AD_flags : pandas.DataFrame
        The updated applicability domain flags DataFrame.
    '''
    where_all_missing = predictions.isna().all(axis=1)
    if any(where_all_missing):
        # Drop chemicals missing all predictions (e.g., inorganics)
        predictions = predictions.loc[~where_all_missing]
        logging.info(f'Dropped {sum(where_all_missing)} rows with all missing predictions')

    where_duplicated_idx = predictions.index.duplicated()
    if any(where_duplicated_idx):
        # Drop duplicate chemicals
        predictions = predictions.loc[~where_duplicated_idx]
        logging.info(f'Dropped {sum(where_duplicated_idx)} duplicated rows')

    # Update the applicability domain flags
    AD_flags = AD_flags.loc[predictions.index]

    return predictions, AD_flags
#endregion

#region: _write_data
def _write_data(df, write_path):
    '''
    Write DataFrame to a Parquet file with gzip compression.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be written to disk.
    write_path : str
        The file path where the DataFrame will be saved. If `write_path` is 
        None or an empty string, the function does nothing.

    Returns
    -------
    None
    '''
    if write_path:
        utilities.ensure_directory_exists(write_path)
        df.to_parquet(write_path, compression='gzip')
#endregion
    
#region: extract_predictions_and_app_domains
def extract_predictions_and_app_domains(
        data_dir, 
        columns_for_model, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log'
        ):
    '''
    Process a single batch of chemicals/samples.

    Returns
    -------
    - Applicability domain flags (pandas.DataFrame)
    - Predictions (pandas.DataFrame)
    '''
    AD_flags = extract_app_domains_from_csv_files(
        data_dir, 
        columns_for_model, 
        index_name=index_name,
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix,  
        log10_pat=log10_pat
    )

    predictions = extract_predictions_from_csv_files(
        data_dir, 
        columns_for_model, 
        index_name=index_name, 
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix, 
        log10_pat=log10_pat, 
        flags=AD_flags
    )
    
    if predictions.index.duplicated().any():
        raise ValueError('Duplicate DTXSIDs found in directory. Check input data.')
    
    return predictions, AD_flags
#endregion

#region: extract_predictions_from_csv_files
def extract_predictions_from_csv_files(
        data_dir, 
        columns_for_model, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        flags=None
        ):
    '''
    Load and extract the outputs as separate CSV file from OPERA 2.9.

    Parameters
    ----------
    data_dir : str
        Path to the data directory. 
    columns_for_model : dict 
        Mapping of OPERA2.9 model names (str) to the desired column names 
        (str) in the data CSV file.
    index_name : str (optional)
        Can be used to rename the default index ('MoleculeID').
    discrete_columns : list of str (optional)
        Column names corresponding to discrete data.
    discrete_suffix : str (optional)
        Will be appended to the end of each string in discrete_columns.
    log10_pat : str (optional)
        Substring in the columns indicating log10-transformed features, will 
        be used to inverse-transform these features.
    flags : pandas.DataFrame (optional)
        Maps each feature-chemical combination to a boolean. Values which 
        are 'True' denote unreliable and will be considered as missing (NaN).
    
    Returns
    -------
    pandas.DataFrame
        Axis 0 = chemical ID; Axis 1 = feature.
    '''
    predictions = []  # initialize
    for entry in os.listdir(data_dir):
        if entry.endswith('.csv'):
            model_name = _extract_model_name_from_file_name(entry)
            if model_name in list(columns_for_model):
                model_data = _model_data_from_csv(data_dir, entry)
                predictions.append(model_data[columns_for_model[model_name]])

    ## Assemble the final DataFrame.

    predictions = pd.concat(predictions, axis=1)

    predictions = predictions[get_original_columns(columns_for_model)]

    if index_name is not None:
        # Rename the 'MoleculeID' column
        predictions.index.name = index_name
    if log10_pat is not None:
        predictions = utilities.inverse_log10_transform(
            predictions, 
            log10_pat
        )
        predictions.columns = utilities.remove_pattern(
            predictions.columns, 
            log10_pat
        )
    if discrete_columns is not None:
        predictions = utilities.tag_discrete_columns(
            predictions, 
            discrete_columns, 
            discrete_suffix
        )
    if flags is not None:
        predictions = set_unreliable_values(predictions, flags)

    return predictions
#endregion

#region: _extract_model_name_from_file_name
def _extract_model_name_from_file_name(file_name):
    '''
    Helper function to extract an OPERA model name from the specified file 
    name.

    This function leverages OPERA's file name convention:
    "<prefix>_<model_name>.csv"

    Parameters
    ----------
    file_name : str

    Returns
    -------
    str
        The model name.
    '''
    return file_name.split('_')[-1].split('.')[0]
#endregion

#region: _model_data_from_csv
def _model_data_from_csv(data_dir, file_name):
    '''
    Helper function to load OPERA model data from a CSV file.

    The first column (MoleculeID) is used as the index column
    '''
    data_path = os.path.join(data_dir, file_name)
    return pd.read_csv(
        data_path, 
        index_col=0, 
        low_memory=False  # silences warnings about mixed dtypes
        )
#endregion

#region: set_unreliable_values
def set_unreliable_values(predictions, AD_flags):
    '''
    Helper function which sets unreliable values in 'predictions' as NaN.

    For safety, the columns of 'AD_flags' must be in 'predictions'.

    See Also
    --------
    common.opera.data_from_csv_files()

    Raises
    ------
    ValueError
        If the columns of 'AD_flags' are not in 'predictions'.
    '''
    predictions = predictions.copy()

    shared_features = AD_flags.columns.intersection(predictions.columns)
    if len(shared_features) != len(AD_flags.columns):
        raise ValueError('The columns of "AD_flags" must be in "predictions"')
        
    for feature in shared_features:
        where_unreliable = AD_flags[feature]
        predictions.loc[where_unreliable, feature] = np.NaN    

    return predictions
#endregion

#region: extract_app_domains_from_csv_files
def extract_app_domains_from_csv_files(
        data_dir, 
        columns_for_model, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat=None
        ):
    '''
    Flag any features outside the respective model applicability domains.

    Returns a DataFrame corresponding to common.opera.data_from_csv_files().
    Each column (feature) is a boolean mask where True denotes that a given
    chemical falls outside the applicability domain and may be unreliable.

    References
    ----------
    https://doi.org/10.1186/s13321-018-0263-1
    '''
    AD_flags = {}   # initialize

    for entry in os.listdir(data_dir):
        if entry.endswith('.csv'):
            model_name = _extract_model_name_from_file_name(entry)
            if model_name in list(columns_for_model):
                model_data = _model_data_from_csv(data_dir, entry)
        
                where_ADs = model_data.columns.str.contains('^AD_')
                ADs = model_data.loc[:, where_ADs]
                
                feature_columns = columns_for_model[model_name]
                if list(ADs):
                    N_ADs = len(ADs.columns)
                    if N_ADs == 2: 
                        # All features share a pair of ADs: 'global' and 'local'.
                        for feature in feature_columns:
                            AD_flags[feature] = (
                                is_outside_applicability_domain(ADs)
                                )
                    elif N_ADs//2 == len(feature_columns):
                        # Each feature has its own pair of ADs.
                        for feature in feature_columns:
                            where_feature_ADs = ADs.columns.str.contains(
                                feature.strip('_pred'))
                            feature_specific_ADs = ADs.loc[:, where_feature_ADs]
                            AD_flags[feature] = (
                                is_outside_applicability_domain(feature_specific_ADs)
                                )

    ## Assemble the final DataFrame.
    AD_flags = pd.DataFrame(AD_flags)

    original_columns = [c for c in get_original_columns(columns_for_model) if c in AD_flags]
    AD_flags = AD_flags[original_columns]

    if index_name is not None:
        # Rename the 'MoleculeID' column
        AD_flags.index.name = index_name
    if log10_pat is not None:
        AD_flags.columns = utilities.remove_pattern(
            AD_flags.columns, 
            log10_pat
        )
    if discrete_columns is not None:
        AD_flags = utilities.tag_discrete_columns(
            AD_flags, 
            discrete_columns, 
            discrete_suffix,
            validate_columns=False  # not all features have a defined AD
            )

    return AD_flags
#endregion

#region: is_outside_applicability_domain
def is_outside_applicability_domain(global_local_ADs):
    '''
    Find chemicals that may be unreliable.

    A chemical may be unreliable if,
        1. outside the model's global applicability domain
        2. low local applicability domain index (< 0.4)

    Parameters
    ----------
    global_local_ADs : pandas.DataFrame
        Pair of applicability domains for a given feature.

    Returns
    -------
    boolean mask

    See Also
    --------
    extract_app_domains_from_csv_files()
    '''
    global_ADs, local_AD_indexes = (
        split_applicability_domain_columns(global_local_ADs))
    where_outside_global_AD = global_ADs == 0.
    where_low_local_AD_index = local_AD_indexes < 0.4
    return where_outside_global_AD & where_low_local_AD_index
#endregion

#region: split_applicability_domain_columns
def split_applicability_domain_columns(global_local_ADs):
    '''
    Split a DataFrame with two columns into its respective Series, one for 
    each type of applicability domain.

    Assumes that the columns follow the naming convention of OPERA 2.9.

    Parameters
    ----------
    global_local_ADs : pandas.DataFrame

    Returns
    -------
    global_ADs : pandas.Series
        Global applicability domains.
    local_AD_indexes : pandas.Series
        Local applicability domain indexes.
    '''
    if len(global_local_ADs.columns) != 2:
        print(global_local_ADs.head())
        raise ValueError(
            '"global_local_ADs" must have exactly two columns: '
            'one global, one local')

    ## Identify the relevant column names.
    global_AD_column = []
    local_AD_column = []
    for col in global_local_ADs:
        if col.startswith('AD_') and 'index' not in col:
            global_AD_column.append(col)
        elif col.startswith('AD_index'):
            local_AD_column.append(col)

    global_ADs = global_local_ADs[global_AD_column].squeeze()
    local_AD_indexes = global_local_ADs[local_AD_column].squeeze()

    return global_ADs, local_AD_indexes
#endregion

#region: extract_dtxsid_from_structures_file
def extract_dtxsid_from_structures_file(structures_file):
    '''
    Extract DTXSID values from a SMI file.

    Parameters
    ----------
    structures_file : str
        The path to the SMI file.

    Returns
    -------
    list
    '''
    return extract_from_smi_file(structures_file, 1)
#endregion

#region: extract_smiles_from_structures_file
def extract_smiles_from_structures_file(structures_file):
    '''
    Extract "QSAR-ready" SMILES strings from a SMI file.

    Parameters
    ----------
    structures_file : str
        The path to the SMI file.

    Returns
    -------
    list
    '''
    return extract_from_smi_file(structures_file, 0)
#endregion

#region: extract_from_smi_file
def extract_from_smi_file(structures_file, index):
    '''
    Helper function to extract data from a SMI file based on the given index.

    Parameters
    ----------
    structures_file : str
        The path to the SMI file.
    index : int
        The index of the data to extract from each line (0 for SMILES, 1 for 
        DTXSID).

    Returns
    -------
    list
    '''
    with open(structures_file, 'r') as f:
        data = [line.split('\t')[index].strip() for line in f.readlines()]
    return data
#endregion