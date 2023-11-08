'''
'''

import streamlit as st
from io import BytesIO
import zipfile

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
    zip_buffer = create_downloadable_zip_file(inputs, config)

    selected_inputs = [v for k, v in inputs.items() if 'selected' in k]
    if any(selected_inputs):
        # Offer the zip file for download
        st.download_button(
            label='Download Selected Datasets',
            data=zip_buffer,
            file_name='datasets.zip',
            mime='application/zip'
        )
    else:
        st.error('Please select at least one dataset in the sidebar.')
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
            pod_csv = pod_data.to_csv()
            # TODO: Could hard code the extensions rather than config
            pod_file_name = config['pod_file_name'].replace('parquet', 'csv')
            zip_file.writestr(pod_file_name, pod_csv)

        if moe_selected:
            moe_data = dm.load_margins_of_exposure(config, effect_label)
            moe_csv = moe_data.to_csv()
            moe_file_name = config['moe_file_name'].replace('parquet', 'csv')
            zip_file.writestr(moe_file_name, moe_csv)

        if features_selected:
            X = dm.load_features(config, effect_label)
            X_csv = X.to_csv()
            X_file_name = config['features_file_name'].replace('parquet', 'csv')
            zip_file.writestr(X_file_name, X_csv)

    # Set the pointer of the BytesIO object to the beginning
    zip_buffer.seek(0)
    return zip_buffer
#endregion

# Execute the page
if __name__ == '__main__':
    main()