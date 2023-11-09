'''
'''

import streamlit as st
from io import BytesIO
import zipfile
from datetime import datetime
import os.path

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
    metadata_content, metadata_file_names = initialize_metadata(
        config['meta_header_file_name'], 
        inputs['effect_label']
        )

    # Create a zip file in memory
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
        if inputs['pod_selected']:
            write_pods_to_zip_file(config, inputs['effect_label'], zip_file)
            metadata_file_names.append(config['pod_meta_file_name'])
        if inputs['moe_selected']:
            write_moes_to_zip_file(config, inputs['effect_label'], zip_file)
            metadata_file_names.append(config['moe_meta_file_name'])
        if inputs['features_selected']:
            write_features_to_zip_file(config, inputs['effect_label'], zip_file)
            metadata_file_names.append(config['features_meta_file_name'])

        metadata_content += append_metadata(
            metadata_file_names, 
            config['metadata_dir']
            )
        zip_file.writestr('README.txt', metadata_content.encode('utf-8'))

    # Set the pointer of the BytesIO object to the beginning
    zip_buffer.seek(0)
    return zip_buffer
#endregion

# TODO: Move to data_management.py?
#region: write_pods_to_zip_file
def write_pods_to_zip_file(config, effect_label, zip_file):
    '''
    '''
    pod_data = dm.load_points_of_departure(config, effect_label)
    pods = pod_data['POD'] 
    write_to_zip_file(pods, config['pod_file_name'], zip_file)
#endregion

#region: write_moes_to_zip_file
def write_moes_to_zip_file(config, effect_label, zip_file):
    '''
    '''
    moe_data = dm.load_margins_of_exposure(config, effect_label)
    moes = moe_data.drop('Cum_Count', axis=1)
    write_to_zip_file(moes, config['moe_file_name'], zip_file)
#endregion

#region: write_features_to_zip_file
def write_features_to_zip_file(config, effect_label, zip_file):
    '''
    '''
    X = dm.load_features(config, effect_label)
    write_to_zip_file(X, config['features_file_name'], zip_file)
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

# Execute the page
if __name__ == '__main__':
    main()