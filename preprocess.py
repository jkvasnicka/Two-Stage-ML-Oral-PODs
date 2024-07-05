'''
This module orchestrates the preprocessing of all raw datasets, including 
features, target variables, and datasets for plotting and analysis. The 
configuration settings are used to identify and process each raw data source 
through the RawDataProcessor's specified methods.
'''

from config_management import config_from_cli_args
from raw_processing.processor import RawDataProcessor

def main():
    '''
    Execute the preprocessing steps for eacg raw data source defined in the
    RawDataProcessor.

    Returns
    -------
    None
        The processed data are written to a dedicated directory.
    '''
    config = config_from_cli_args()

    raw_processor = RawDataProcessor(config.raw_data, config.data, config.path)

    for k, process_from_raw in raw_processor.dispatcher.items():
        print(f'\t{k}...')
        process_from_raw()

if __name__ == '__main__':
    print('Preprocessing from raw...')
    main()
    print('Preprocessing completed.')
