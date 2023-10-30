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
    def effect_label(self):
        '''
        '''
        return self._effect_label
    
    @effect_label.setter
    def effect_label(self, new_effect_label):
        '''
        '''
        self._effect_label =  new_effect_label

        if new_effect_label:
            # Get the corresponding key for data lookups
            self._effect = self._config['effect_for_label'][new_effect_label]
            # Append the effect name to the data directory
            self._effect_subdir = os.path.join(
                self._config['data_dir'], 
                self._effect
            )
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
        smiles_for_id = load_qsar_ready_smiles(
            self._config['data_dir'], 
            self._config['chem_ids_file_name']
            )
        
        return smiles_for_id[chemical_id]
    #endregion

    #region: features
    @property
    def features(self):
        '''
        '''
        X = load_features(
            self._effect_subdir,
            self._config['features_file_name']
        )

        return X.loc[self.chemical_id]
    #endregion

    #region: point_of_departure
    @property
    def point_of_departure(self):
        '''
        '''
        pod_data = load_points_of_departure(
            self._effect_subdir,
            self._config['pod_file_name']
        )

        return pod_data.loc[self.chemical_id]
    #endregion

#region: load_qsar_ready_smiles
@st.cache_data
def load_qsar_ready_smiles(effect_subdir, file_name):
    '''
    '''
    smiles_for_id = (
        read_parquet(
            effect_subdir,
            file_name
        )
        .squeeze()
        .to_dict()
    )

    return smiles_for_id
#endregion

#region: load_features
@st.cache_data
def load_features(effect_subdir, file_name):
    '''
    '''
    return read_parquet(effect_subdir, file_name)
#endregion

#region: load_points_of_departure
@st.cache_data
def load_points_of_departure(effect_subdir, file_name):
    '''
    '''
    return read_parquet(effect_subdir, file_name)
#endregion

#region: read_parquet
def read_parquet(effect_subdir, file_name):
    '''
    '''
    full_path = os.path.join(effect_subdir, file_name)
    return pd.read_parquet(full_path)
#endregion