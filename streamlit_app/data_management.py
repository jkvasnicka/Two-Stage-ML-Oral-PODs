'''
'''

import streamlit as st
import os
import pandas as pd 
import json 

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

#region: read_data
def read_data(data_file):
    '''
    '''
    extension = data_file.split('.')[-1]

    if extension == 'parquet':
        return pd.read_parquet(data_file)
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