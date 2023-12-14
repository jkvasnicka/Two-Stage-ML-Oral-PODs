'''This module contains functions for loading/parsing raw data from OPERA 2.9.

The OPERA model was run with the following output options:
    - Separate files
    - Experimental values
    - Nearest neighbors
    - INclude descriptor values

Source: https://github.com/NIEHS/OPERA/releases/tag/v2.9.1
'''

import pandas as pd 
import numpy as np
import os
import logging
import json

from . import utilities

#region: process_all_batches
def process_all_batches(
        main_dir, 
        columns_mapper_path, 
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

    Parameters
    ----------
    main_dir : str
        The path to the main directory.

    Returns
    -------
    pd.DataFrame
        A DataFrame with combined data from all processed directories.
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
                        columns_mapper_path, 
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

    if data_write_path is not None:
        predictions.to_csv(data_write_path)
    if flags_write_path is not None:
        AD_flags.to_csv(flags_write_path)

    return predictions, AD_flags
#endregion
    
#region: extract_predictions_and_app_domains
def extract_predictions_and_app_domains(
        data_dir, 
        columns_mapper_path, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        data_write_path=None, 
        flags_write_path=None
        ):
    '''
    '''
    AD_flags = extract_app_domains_from_csv_files(
        data_dir, 
        columns_mapper_path, 
        index_name=index_name,
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix,  
        log10_pat=log10_pat, 
        write_path=flags_write_path
    )

    predictions = extract_predictions_from_csv_files(
        data_dir, 
        columns_mapper_path, 
        index_name=index_name, 
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix, 
        log10_pat=log10_pat, 
        flags=AD_flags,
        write_path=data_write_path
        )
    
    if predictions.index.duplicated().any():
        raise ValueError('Duplicate DTXSIDs found in directory. Check input data.')
    
    return predictions, AD_flags
#endregion

#region: extract_predictions_from_csv_files
def extract_predictions_from_csv_files(
        data_dir, 
        columns_mapper_path, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        flags=None, 
        write_path=None
        ):
    '''Load and extract the outputs as separate CSV file from OPERA 2.9.

    Parameters
    ----------
    data_dir : str
        Path to the data directory. 
    columns_mapper_path : str 
        Path to a JSON file that maps model names to the desired column 
        names in the data CSV file: mapper[model_name] --> list of str.
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
    write_path str (optional)
        Path to write the return as a CSV file.
    
    Returns
    -------
    pandas.DataFrame
        Axis 0 = chemical ID; Axis 1 = feature.
    '''
    columns_for_model = json_to_dict(columns_mapper_path)

    predictions = []  # initialize
    for entry in os.listdir(data_dir):
        if entry.endswith('.csv'):
            model_name = _extract_model_name_from_file_name(entry)
            if model_name in list(columns_for_model):
                model_data = _model_data_from_csv(data_dir, entry)
                predictions.append(model_data[columns_for_model[model_name]])

    ## Assemble the final DataFrame.

    predictions = pd.concat(predictions, axis=1)
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
    if write_path is not None:
        predictions.to_csv(write_path)

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
    return pd.read_csv(data_path, index_col=0)
#endregion

#region: set_unreliable_values
def set_unreliable_values(predictions, AD_flags):
    '''Helper function which sets unreliable values in 'predictions' as NaN.

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
        columns_mapper_path, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat=None, 
        write_path=None
        ):
    '''Flag any features outside the respective model applicability domains.

    Returns a DataFrame corresponding to common.opera.data_from_csv_files().
    Each column (feature) is a boolean mask where True denotes that a given
    chemical falls outside the applicability domain and may be unreliable.

    References
    ----------
    https://doi.org/10.1186/s13321-018-0263-1
    '''
    columns_for_model = json_to_dict(columns_mapper_path)

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
    if write_path is not None:
        AD_flags.to_csv(write_path)

    return AD_flags
#endregion

#region: is_outside_applicability_domain
def is_outside_applicability_domain(global_local_ADs):
    '''Find chemicals that may be unreliable.

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

#region: chemicals_to_exclude_from_qsar
def chemicals_to_exclude_from_qsar(
        chemical_id_file, chemical_structures_file):
    '''Return a list of chemicals that did not pass the QSAR Standardization 
    Workflow.
    '''
    raw_chemical_ids = set(pd.read_csv(chemical_id_file).squeeze())
    qsar_ready_ids = set(
        extract_dtxsid_from_structures_file(chemical_structures_file)
    )

    return list(raw_chemical_ids.difference(qsar_ready_ids))
#endregion

#region: extract_dtxsid_from_structures_file
def extract_dtxsid_from_structures_file(structures_file):
    '''
    Extracts DTXSID values from a SMI file.

    Parameters
    ----------
    structures_file : str
        The path to the SMI file.
    index_col : str
        The name of the index column in the output Series.

    Returns
    -------
    pd.Series
        A Series with the DTXSID values.
    '''
    with open(structures_file, 'r') as f:
        # structure, dtxsid = line.split('\t')
        dtxsid = [line.split('\t')[1].strip() for line in f.readlines()]
    return dtxsid
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
        return json.loads(f.read())
#endregion