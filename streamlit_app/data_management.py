'''
'''

import streamlit as st
import os
import pandas as pd 
import json 

#region: DataManager.__init__
class DataManager:
    def __init__(self, config_file='config.json'):
        '''
        '''
        with open(config_file, 'r') as file:
            self._config = json.load(file)

        self.chemical_id = '' 
        self.effect_label = ''
#endregion

    #region: effect_labels
    @property
    def effect_labels(self):
        '''
        '''
        return list(self._config['effect_for_label'])
    #endregion

    #region: effect 
    @property
    def effect(self):
        '''
        '''
        return self._config['effect_for_label'][self.effect_label]
    #endregion

    #region: chemical_id 
    @property
    def chemical_id(self):
        '''
        '''
        return self._chemical_id

    @chemical_id.setter
    def chemical_id(self, new_id):
        '''
        '''
        self._chemical_id = new_id

        if new_id:
            self.qsar_ready_smiles = self.get_qsar_ready_smiles(new_id)
    #endregion

    #region: get_qsar_ready_smiles
    def get_qsar_ready_smiles(self, chemical_id):
        '''
        '''
        smiles_file = self.build_data_path(
            'chem_ids_file_name',
            is_effect_specific=False
        )
        smiles_for_id = load_qsar_ready_smiles(smiles_file)
        
        return smiles_for_id[chemical_id]
    #endregion

    #region: features
    @property
    def features(self):
        '''
        '''
        features_file = self.build_data_path('features_file_name')
        X = load_features(features_file)

        return X.loc[self.chemical_id]
    #endregion

    # TODO: Change to 'points_of_departure'
    #region: point_of_departure
    @property
    def point_of_departure(self):
        '''
        '''
        pod_file = self.build_data_path('pod_file_name')
        pod_data = load_points_of_departure(pod_file)

        return pod_data.loc[self.chemical_id]
    #endregion

    #region: margins_of_exposure
    @property
    def margins_of_exposure(self):
        '''
        '''
        moe_file = self.build_data_path('moe_file_name')
        moe_data = load_margins_of_exposure(moe_file)

        return moe_data.loc[self.chemical_id]
    #endregion

    #region: build_data_path
    def build_data_path(self, file_key, is_effect_specific=True):
        '''
        '''
        file_name = self._config[file_key]

        data_dir = self._config['data_dir']
        if is_effect_specific:
            data_dir = os.path.join(data_dir, self.effect)
        
        return os.path.join(data_dir, file_name)
    #endregion

#region: load_qsar_ready_smiles
@st.cache_data
def load_qsar_ready_smiles(smiles_file):
    '''
    '''
    smiles_for_id = (
        read_data(smiles_file)
        .squeeze()
        .to_dict()
    )

    return smiles_for_id
#endregion

#region: load_features
@st.cache_data
def load_features(features_file):
    '''
    '''
    return read_data(features_file)
#endregion

#region: load_points_of_departure
@st.cache_data
def load_points_of_departure(pod_file):
    '''
    '''
    return read_data(pod_file)
#endregion

#region: load_margins_of_exposure
@st.cache_data
def load_margins_of_exposure(moe_file):
    '''
    '''
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