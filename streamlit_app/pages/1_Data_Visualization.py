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

        effect_label, chemical_id = get_user_inputs(effect_labels)

    if is_valid_user_input(chemical_id):
        render_outputs(config, effect_label, chemical_id)
#endregion

# TODO: Derive parameters from config 
#region: initialize_page
def initialize_page(config):
    '''
    '''
    st.set_page_config(
        page_title='Data Visualization',
        layout='wide'
    )

    st.title('Interactive Data Visualization Tool')
#endregion

#region: get_user_inputs
def get_user_inputs(effect_labels):
    '''
    '''
    effect_label = st.selectbox(
        'Select the effect category',
        options=effect_labels
    )

    chemical_id = st.text_input(
        'Enter the DTXSID for the chemical',
        placeholder='e.g., DTXSID2021315'
        )

    return effect_label, chemical_id
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
        effect_label,
        chemical_id 
        ):
    '''
    '''
    if chemical_id:

        with st.sidebar:
            smiles_for_id = dm.load_qsar_ready_smiles(config)
            render.qsar_ready_structure(smiles_for_id[chemical_id])

        if effect_label:
                
            # Initialize a grid with 2 rows and 2 columns
            grid = [
                st.columns(2), 
                st.columns(2)
            ]

            with grid[0][1]:  # top right
                pod_data = dm.load_points_of_departure(config, effect_label)
                render.points_of_departure(pod_data.loc[chemical_id])
            with grid[0][0]:  # top left
                pod_fig = dm.load_pod_figure(config, effect_label)
                render.pod_figure(
                    pod_fig, 
                    pod_data.loc[chemical_id]
                )
            with grid[1][1]:  # bottom right
                moe_data = dm.load_margins_of_exposure(config, effect_label)
                render.margins_of_exposure(moe_data.loc[chemical_id], config)
            with grid[1][0]:  # bottom left
                moe_fig = dm.load_moe_figure(config, effect_label)
                render.moe_figure(
                    moe_fig, 
                    moe_data.loc[chemical_id]
                )
#endregion

# Call the main function to execute the app
if __name__ == '__main__':
    main()