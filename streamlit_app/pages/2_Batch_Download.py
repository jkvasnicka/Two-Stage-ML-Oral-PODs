'''
'''

import streamlit as st

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
        effect_label = get_user_inputs(effect_labels)
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
    effect_label = st.selectbox(
        'Select the effect category',
        options=effect_labels
    )

    return effect_label
#endregion

# Execute the page
if __name__ == '__main__':
    main()