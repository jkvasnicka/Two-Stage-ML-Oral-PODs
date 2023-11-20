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
        Download the datasets used by the app.
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
    metadata_content, metadata_file_names = dm.initialize_metadata(
        config['meta_header_file_name'], 
        inputs['effect_label']
        )

    # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:

        if inputs['pod_selected']:
            dm.write_pods_to_zip_file(config, inputs['effect_label'], zip_file)
            metadata_file_names.append(config['pod_meta_file_name'])
        if inputs['moe_selected']:
            dm.write_moes_to_zip_file(config, inputs['effect_label'], zip_file)
            metadata_file_names.append(config['moe_meta_file_name'])
        metadata_content += dm.append_metadata(
            metadata_file_names, 
            data_dir=config['metadata_dir']
            )
        
        # Handle feature descriptions separately
        if inputs['features_selected']:
            feature_names = dm.write_features_to_zip_file(
                config, 
                inputs['effect_label'], 
                zip_file
                )
            metadata_content += dm.append_metadata(
                [config['features_meta_file_name']], 
                data_dir=config['metadata_dir']
            )
            metadata_content += dm.extract_feature_descriptions(
                config['features_desc_file_name'], 
                data_dir=config['metadata_dir'],
                feature_names=feature_names
            )

        zip_file.writestr('README.txt', metadata_content.encode('utf-8'))

    # Set the pointer of the BytesIO object to the beginning
    zip_buffer.seek(0)
    return zip_buffer
#endregion

# Execute the page
if __name__ == '__main__':
    main()