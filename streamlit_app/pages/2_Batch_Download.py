'''
'''

import streamlit as st
from io import BytesIO
import zipfile
from datetime import datetime

import data_management as dm

#region: main
def main():
    '''
    '''
    config = dm.load_config()

    initialize_page(config)

    with st.sidebar:
        st.header('User Input')
        effect_labels = dm.get_effect_labels(config)
        inputs = get_user_inputs(effect_labels)

    prepare_data_download(inputs, config) 
#endregion

#region: initialize_page
def initialize_page(config):
    '''
    '''
    st.set_page_config(
        page_title='Download',
        layout='wide'
    )

    st.title('Batch Data Download')

    st.markdown(
        '''
        Download data for all chemicals, or for a subset of all chemicals.
        '''
    )
#endregion

#region: get_user_inputs
def get_user_inputs(effect_labels):
    '''
    '''
    inputs = {}  # initialize

    inputs['effect_label'] = st.selectbox(
        'Select the effect category',
        options=effect_labels
    )

    inputs['pod_selected'] = st.checkbox('Points of Departure')
    inputs['moe_selected'] = st.checkbox('Margins of Exposure')
    inputs['features_selected'] = st.checkbox('Features (OPERA 2.9)')

    return inputs
#endregion

#region: prepare_data_download
def prepare_data_download(inputs, config):
    '''
    '''
    selected_inputs = [v for k, v in inputs.items() if 'selected' in k]
    
    if any(selected_inputs):

        with st.spinner('Preparing the data for download...'):
            zip_buffer = create_downloadable_zip_file(inputs, config)

        # Offer the zip file for download
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f'ML2_Data_Export_{timestamp}.zip'
        st.download_button(
            label='Download Selected Datasets',
            data=zip_buffer,
            file_name=zip_filename,
            mime='application/zip'
        )

    else:
        st.info(':point_left: Please select at least one dataset in the sidebar.')
#endregion

#region: create_downloadable_zip_file
def create_downloadable_zip_file(inputs, config):
    '''
    '''
    effect_label = inputs['effect_label']
    pod_selected = inputs['pod_selected']
    moe_selected = inputs['moe_selected']
    features_selected = inputs['features_selected']

    # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        
        if pod_selected:
            pod_data = dm.load_points_of_departure(config, effect_label)
            write_to_zip_file(pod_data, config['pod_file_name'], zip_file)

        if moe_selected:
            moe_data = dm.load_margins_of_exposure(config, effect_label)
            write_to_zip_file(moe_data, config['moe_file_name'], zip_file)

        if features_selected:
            X = dm.load_features(config, effect_label)
            write_to_zip_file(X, config['features_file_name'], zip_file)

    # Set the pointer of the BytesIO object to the beginning
    zip_buffer.seek(0)
    return zip_buffer
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

# Execute the page
if __name__ == '__main__':
    main()