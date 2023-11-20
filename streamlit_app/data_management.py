'''
'''

import streamlit as st
import os
import pandas as pd 
import json 
import pickle
import gzip

#region: load_config
def load_config(config_file='config.json'):
    '''
    '''
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config
#endregion

#region: get_effect_labels
def get_effect_labels(config):
    '''
    '''
    return list(config['effect_for_label'])
#endregion

#region: load_qsar_ready_smiles
@st.cache_data
def load_qsar_ready_smiles(config):
    '''
    '''
    smiles_file = build_data_path('chem_ids_file_name', config)

    smiles_for_id = (
        read_data(smiles_file)
        .squeeze()
        .to_dict()
    )

    return smiles_for_id
#endregion

#region: load_features
@st.cache_data
def load_features(config, effect_label):
    '''
    '''
    features_file = build_data_path(
        'features_file_name',
        config,
        effect_label=effect_label
    )
    
    return read_data(features_file)
#endregion

#region: load_points_of_departure
@st.cache_data
def load_points_of_departure(config, effect_label):
    '''
    '''
    pod_file = build_data_path(
        'pod_file_name',
        config, 
        effect_label=effect_label
    )

    return read_data(pod_file)
#endregion

#region: load_margins_of_exposure
@st.cache_data
def load_margins_of_exposure(config, effect_label):
    '''
    '''
    moe_file = build_data_path(
        'moe_file_name',
        config, 
        effect_label=effect_label
    )

    return read_data(moe_file)
#endregion

#region: load_pod_figure
@st.cache_data
def load_pod_figure(config, effect_label):
    '''
    '''
    pod_fig_file = build_data_path(
        'pod_fig_file_name',
        config, 
        effect_label=effect_label
    )

    return read_data(pod_fig_file)
#endregion

#region: load_moe_figure
@st.cache_data
def load_moe_figure(config, effect_label):
    '''
    '''
    moe_fig_file = build_data_path(
        'moe_fig_file_name',
        config, 
        effect_label=effect_label
    )

    return read_data(moe_fig_file)
#endregion

#region: read_data
def read_data(data_file):
    '''
    '''
    extension = data_file.split('.')[-1]

    if extension == 'parquet':
        return pd.read_parquet(data_file)
    
    elif extension == 'pkl':
        with gzip.open(data_file, 'rb') as pkl_file:
            object = pickle.load(pkl_file)
        return object
    
    else:
        raise ValueError(f'File extension not accepted: {extension}')
#endregion

#region: build_data_path
def build_data_path(file_key, config, effect_label=None):
    '''
    '''
    data_dir = config['data_dir']
    file_name = config[file_key]

    if effect_label:
        effect_subdir = config['effect_for_label'][effect_label]
    else:
        effect_subdir = ''
        
    return os.path.join(data_dir, effect_subdir, file_name)
#endregion

#region: write_pods_to_zip_file
def write_pods_to_zip_file(config, effect_label, zip_file):
    '''
    '''
    pod_data = load_points_of_departure(config, effect_label)
    pods = pod_data['POD'] 
    write_to_zip_file(pods, config['pod_file_name'], zip_file)
#endregion

#region: write_moes_to_zip_file
def write_moes_to_zip_file(config, effect_label, zip_file):
    '''
    '''
    moe_data = load_margins_of_exposure(config, effect_label)
    moes = moe_data.drop('Cum_Count', axis=1)
    write_to_zip_file(moes, config['moe_file_name'], zip_file)
#endregion

#region: write_features_to_zip_file
def write_features_to_zip_file(config, effect_label, zip_file):
    '''
    '''
    X = load_features(config, effect_label)
    write_to_zip_file(X, config['features_file_name'], zip_file)
    feature_names = list(X)
    return feature_names
#endregion

#region: write_to_zip_file
def write_to_zip_file(data, file_name, zip_file):
    '''
    '''
    csv_data = data.to_csv()
    # Remove any previous extension
    file_name = file_name.split('.')[0] + '.csv'    
    zip_file.writestr(file_name, csv_data)
#endregion

#region: initialize_metadata
def initialize_metadata(meta_header_file_name, effect_label):
    '''
    '''
    metadata_content = f'Downloaded Datasets for Effect Category, "{effect_label}"\n'
    metadata_content += '=' * len(metadata_content) + '\n\n'

    metadata_file_names = []
    metadata_file_names.append(meta_header_file_name)
    
    return metadata_content, metadata_file_names
#endregion

#region: append_metadata
def append_metadata(metadata_file_names, data_dir=''):
    '''
    '''
    metadata_content = ''  # initialize
    for file_name in metadata_file_names:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'r') as file:
            metadata_content += file.read() + '\n\n'
    return metadata_content
#endregion

#region: append_metadata_to_zip
def append_metadata_to_zip(zip_file, metadata_content, file_name):
    '''
    '''
    zip_file.writestr(file_name, metadata_content.encode('utf-8'))
#endregion

#region: extract_feature_descriptions
def extract_feature_descriptions(file_name, data_dir='', feature_names=None):
    '''
    Extract feature descriptions from an Excel file.

    This function assumes that the Excel file has a column 'Feature_Name' 
    which is used as the index column. It also assumes that 'Unit' and 
    'Description' columns are present for each feature. If a list of 
    `feature_names` is provided, only those features will be included in the 
    output. Missing values in 'Unit' or 'Description' will result in an 
    incomplete output for that feature.
    
    Parameters
    ----------
    excel_path : str
        The file path to the Excel file containing the features.
    feature_names : list of str, optional
        A list of feature names to filter the output. If not provided, all 
        features are included.

    Returns
    -------
    str
        A formatted string containing the feature descriptions.
    '''
    excel_path = os.path.join(data_dir, file_name)
    df = pd.read_excel(excel_path, index_col='Feature_Name')
    if feature_names:
        df = df.loc[feature_names]
    
    # Initialize the string
    feature_descriptions = 'Feature Descriptions'
    # Underline the heading
    feature_descriptions += '\n' + '-'*len(feature_descriptions) + '\n\n'
    for feature_name, row in df.iterrows():
        # Write feature name (units) : <description>
        feature_descriptions += (
            f"- {feature_name} ({row['Unit']}) : {row['Description']}.\n"
        )
    return feature_descriptions
#endregion