'''
'''

import streamlit as st 
import re 

import data_management as dm
import render

#region: main
def main():
    '''
    '''        
    config = dm.load_config()

    initialize_page(config)

    with st.sidebar:
        st.header('User Input')

        effect_labels = dm.get_effect_labels(config)

        chemical_id, effect_label = get_user_inputs(effect_labels)

    if is_valid_user_input(chemical_id):
        render_outputs(config, chemical_id, effect_label)
#endregion

# TODO: Derive parameters from config 
#region: initialize_page
def initialize_page(config):
    '''
    '''
    st.set_page_config(
        page_title='Chemical Prediction',
        layout='wide'
    )

    st.title('Chemical Prediction Tool')
#endregion

#region: get_user_inputs
def get_user_inputs(effect_labels):
    '''
    '''
    chemical_id = st.text_input('Enter the DTXSID for the chemical')
    effect_label = st.selectbox(
        'Select the effect category',
        options=effect_labels
    )

    return chemical_id, effect_label
#endregion

# TODO: Test on the entire DSSTox
#region: is_valid_user_input
def is_valid_user_input(chemical_id):
    '''
    '''
    if not chemical_id:
        st.error('Please enter a valid DTXSID in the sidebar.')
        is_valid = False
    elif not is_valid_dtxsid(chemical_id):
        st.error(
            f'The DTXSID, "{chemical_id}",  is invalid. It should be "DTXSID" followed by digits.')
        is_valid = False
    else:
        is_valid = True
    
    return is_valid
#endregion

#region: is_valid_dtxsid
def is_valid_dtxsid(chemical_id):
    '''
    '''
    pattern = r'^DTXSID\d{6,14}$'
    return re.match(pattern, chemical_id) is not None
#endregion

#region: render_outputs
def render_outputs(
        config,
        chemical_id, 
        effect_label
        ):
    '''
    '''
    if chemical_id:
        smiles_for_id = dm.load_qsar_ready_smiles(config)
        render.qsar_ready_structure(smiles_for_id[chemical_id])

        if effect_label:
            X = dm.load_features(config, effect_label)
            render.features(X.loc[chemical_id])
                
            pods = dm.load_points_of_departure(config, effect_label)
            render.points_of_departure(pods.loc[chemical_id])

            moe_data = dm.load_margins_of_exposure(config, effect_label)
            render.margins_of_exposure(moe_data.loc[chemical_id])
#endregion

# Call the main function to execute the app
if __name__ == '__main__':
    main()