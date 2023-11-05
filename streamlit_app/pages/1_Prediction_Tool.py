'''
'''

import streamlit as st 

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

        chemical_id, effect_label, predict_button = get_user_inputs(effect_labels)

    render_outputs(config, chemical_id, effect_label, predict_button)
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
    predict_button = st.button('Get POD Estimates')

    return chemical_id, effect_label, predict_button
#endregion

#region: render_outputs
def render_outputs(
        config,
        chemical_id, 
        effect_label, 
        predict_button
        ):
    '''
    '''
    if chemical_id:

        smiles_for_id = dm.load_qsar_ready_smiles(config)

        col1, col2 = st.columns(2)
        
        with col1:  # left column
            render.qsar_ready_structure(smiles_for_id[chemical_id])

        with col2:  # right column
            if effect_label:
                X = dm.load_features(config, effect_label)
                render.features(X.loc[chemical_id])

    if predict_button:
        pods = dm.load_points_of_departure(config, effect_label)
        render.points_of_departure(pods.loc[chemical_id])
#endregion

# Call the main function to execute the app
if __name__ == '__main__':
    main()