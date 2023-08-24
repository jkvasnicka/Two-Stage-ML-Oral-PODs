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
import re
import json

from features import inverse_log10_transform

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

#region: process_all_batches
def process_all_batches(
        main_dir, 
        columns_mapper_path, 
        data_file_namer, 
        structures_file_name, 
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
    AD_flags, opera_data = [], []

    for dir_name in os.listdir(main_dir):
        logging.info(f"Processing directory: {dir_name}")
        data_dir = os.path.join(main_dir, dir_name)
        structures_file = os.path.join(data_dir, structures_file_name)

        if os.path.isdir(data_dir):
            try:
                AD_flags_df, opera_data_df = parse_data_single_batch(
                    data_dir, 
                    columns_mapper_path, 
                    data_file_namer,
                    structures_file, 
                    index_name=index_name, 
                    discrete_columns=discrete_columns, 
                    discrete_suffix=discrete_suffix, 
                    log10_pat=log10_pat
                    )
                AD_flags.append(AD_flags_df)
                opera_data.append(opera_data_df)
            except Exception as e:
                logging.error(
                    f"Skipping directory {dir_name} due to error: {str(e)}")
                continue
    AD_flags = pd.concat(AD_flags)
    opera_data = pd.concat(opera_data)

    if flags_write_path is not None:
        AD_flags.to_csv(flags_write_path)
    if data_write_path is not None:
        opera_data.to_csv(data_write_path)

    return AD_flags, opera_data
#endregion

# TODO: Could set dtypes for discrete_columns (int).
#region: parse_data_single_batch
def parse_data_single_batch(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        structures_file,
        index_name, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        data_write_path=None, 
        flags_write_path=None
        ):
    '''
    Handles each directory by extracting DTXSID values and calling OPERA 
    functions.

    Returns
    -------
    pd.DataFrame
        A DataFrame with data from OPERA.
    '''
    molecule_ids = extract_dtxsid_from_structures_file(structures_file)
    molecule_ids = pd.Series(molecule_ids, name=index_name)

    # Remove duplicates, if any, and log the event
    if molecule_ids.duplicated().any():
        logging.warning(
            f'Duplicate DTXSIDs found in directory {data_dir}.'
            'Removing duplicates.')
        molecule_ids = molecule_ids.drop_duplicates()

    return parse_data_with_applicability_domains(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        index_name=index_name, 
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix, 
        log10_pat=log10_pat, 
        molecule_ids=molecule_ids, 
        data_write_path=data_write_path, 
        flags_write_path=flags_write_path
        )
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

#region: parse_data_with_applicability_domains
def parse_data_with_applicability_domains(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        molecule_ids=None, 
        data_write_path=None, 
        flags_write_path=None
        ):
    '''
    '''
    AD_flags = applicability_domain_flags(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        index_name=index_name,
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix,  
        log10_pat=log10_pat, 
        molecule_ids=molecule_ids, 
        write_path=flags_write_path
    )

    opera_data = parse_data_from_csv_files(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        index_name=index_name, 
        discrete_columns=discrete_columns, 
        discrete_suffix=discrete_suffix, 
        log10_pat=log10_pat, 
        flags=AD_flags,
        molecule_ids=molecule_ids, 
        write_path=data_write_path
        )
    
    return AD_flags, opera_data
#endregion

#region: parse_data_from_csv_files
def parse_data_from_csv_files(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat='Log', 
        flags=None, 
        molecule_ids=None, 
        write_path=None
        ):
    '''Load and parse the outputs as separate CSV file from OPERA 2.9.

    Parameters
    ----------
    data_dir : str
        Path to the data directory. 
    columns_mapper_path : str 
        Path to a JSON file that maps model names to the desired column 
        names in the data CSV file: mapper[model_name] --> list of str.
    data_file_namer : function
        Inputs each model name from the config_file keys and returns the
        corresponding filenames.
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

    # Initialize a container.
    data_for_model = {}
    for model_name, data_columns in columns_for_model.items():
        file_path = os.path.join(data_dir, data_file_namer(model_name))
        model_data = pd.read_csv(file_path, index_col=0)
        if molecule_ids is not None:
            model_data = model_data.reset_index()
            model_data.index = molecule_ids
        data_for_model[model_name] = model_data[data_columns]

    ## Assemble the final DataFrame.
    data_for_model = pd.concat(data_for_model.values(), axis=1)
    if index_name is not None:
        data_for_model.index.name = index_name
    if log10_pat is not None:
        data_for_model = inverse_log10_transform(data_for_model, log10_pat)
    if discrete_columns is not None:
        data_for_model = rename_discrete_columns(
            data_for_model, discrete_columns, discrete_suffix)
    if flags is not None:
        data_for_model = set_unreliable_values(data_for_model, flags)
    if write_path is not None:
        data_for_model.to_csv(write_path)

    return data_for_model
#endregion

# TODO: Add a check to ensure correct spelling.
#region: rename_discrete_columns
def rename_discrete_columns(
        data_for_model, discrete_columns, suffix):
    '''Tag discrete columns by adding a suffix.

    discrete_columns must be contained in data_for_model.columns.
    '''
    mapper = {col : col+suffix for col in discrete_columns}
    return data_for_model.rename(mapper, axis=1)
#endregion

#region: set_unreliable_values
def set_unreliable_values(data, AD_flags):
    '''Helper function which sets unreliable values in 'data' as NaN.

    For safety, the columns of 'AD_flags' must be in 'data'.

    See Also
    --------
    common.opera.data_from_csv_files()

    Raises
    ------
    ValueError
        If the columns of 'AD_flags' are not in 'data'.
    '''
    data = data.copy()

    shared_features = AD_flags.columns.intersection(data.columns)
    if len(shared_features) != len(AD_flags.columns):
        raise ValueError('The columns of "AD_flags" must be in "data"')
        
    for feature in shared_features:
        where_unreliable = AD_flags[feature]
        data.loc[where_unreliable, feature] = np.NaN    

    return data
#endregion

#region: applicability_domain_flags
def applicability_domain_flags(
        data_dir, 
        columns_mapper_path, 
        data_file_namer, 
        index_name=None, 
        discrete_columns=None, 
        discrete_suffix=None, 
        log10_pat=None, 
        molecule_ids=None, 
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

    # Initialize a container.
    AD_flags = {}

    for model_name, feature_columns in columns_for_model.items():
        
        file_path = os.path.join(data_dir, data_file_namer(model_name))
        model_data = pd.read_csv(file_path, index_col=0)
        if molecule_ids is not None:
            model_data = model_data.reset_index()
            model_data.index = molecule_ids
        
        where_ADs = model_data.columns.str.contains('^AD_')
        ADs = model_data.loc[:, where_ADs]
        
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
        AD_flags.index.name = index_name
    if log10_pat is not None:
        AD_flags.columns = AD_flags.columns.str.replace(log10_pat, '')
    if discrete_columns is not None:
        AD_flags = rename_discrete_columns(
            AD_flags, discrete_columns, discrete_suffix)
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
    common.opera.applicability_domain_flags()
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

#region: get_failed_directories
def get_failed_directories(main_dir, log_file_name):
    '''
    Get a list of directories with failed operations from the main log file.

    Parameters
    ----------
    main_dir : str
        The main directory containing the main log file.
    log_file_name : str
        The name of the main log file.

    Returns
    -------
    list
        List of directories with failed operations.
    '''
    failed_directories = []
    main_log_file_path = os.path.join(main_dir, log_file_name)

    try:
        with open(main_log_file_path, 'r') as file:
            for line in file:
                if "Skipping directory" in line:
                    dir_name = re.search(
                        'Skipping directory (.*?) due to error', line).group(1)
                    failed_directories.append(dir_name)
    except Exception as e:
        print(f"Could not open main log file due to error: {str(e)}")

    return failed_directories
#endregion

#region: print_error_log_files
def print_error_log_files(
        main_dir, failed_directories, log_file_name="errorLogBatchRun.txt"):
    '''
    Print the contents of the log files located at the given paths.

    Parameters
    ----------
    main_dir : str
        The main directory.
    failed_directories : list
        List of directories with failed operations.
    log_file_name : str, optional
        The name of the log file. Defaults to "errorLogBatchRun.txt".
    '''
    for dir_name in failed_directories:
        path = os.path.join(main_dir, dir_name, log_file_name)
        try:
            with open(path, 'r') as file:
                print(f"Contents of log file at {path}:")
                print(file.read())
                print("\n----------\n")
        except Exception as e:
            print(f"Could not open and print the log file at {path} due to error: {str(e)}")
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