'''
This module contains the `UnifiedConfiguration` class, which centralizes the 
management and access of configuration settings related to different categories,
such as paths, models, and plotting. By unifying the configuration into a 
single object, it allows for a more streamlined access to settings throughout 
the code.

Example
-------
config_files_dict = {
    'path': 'path-configuration.json',
    'model': 'model-configuration.json',
    'plotting': 'plotting-configuration.json'
}
config = UnifiedConfiguration(config_files_dict)
model_settings = config.model
'''

import json
from types import SimpleNamespace
import argparse

#region: UnifiedConfiguration.__init__
class UnifiedConfiguration:
    '''
    This class provides a unified interface to access configuration settings 
    related to different categories such as paths, models, and plotting. The 
    configuration files for each category are loaded and made accessible as 
    attributes.
    '''
    VALID_CATEGORIES = {
        'path',
        'preprocessor',
        'raw_data',
        'data',
        'feature_selection',
        'estimator', 
        'metric', 
        'model',
        'evaluation',
        'plot'
        }

    def __init__(self, config_file=None, encoding=None):
        '''
        Initialize the UnifiedConfiguration object.

        Parameters
        ----------
        config_file : str, optional
            Path to the JSON file mapping categories to configuration file 
            paths. By default, will look for 'config.json' in the working 
            directory.
        encoding : str, optional
            Default is 'utf-8'.
        '''
        # Set defaults
        if config_file is None:
            config_file = 'config.json'  
        if encoding is None:
            encoding = 'utf-8'

        with open(config_file, 'r', encoding=encoding) as mapping_file:
            config_files_dict = json.load(mapping_file)

        # Validate the input categories
        input_categories = set(config_files_dict)
        difference = input_categories - self.VALID_CATEGORIES
        if difference:
            raise ValueError(
                f'Invalid categories {difference}.\n'
                f'Expected: {self.VALID_CATEGORIES}')

        # Load each file into its category
        for category, file_path in config_files_dict.items():
            with open(file_path, 'r', encoding=encoding) as config_file:
                config_dict = json.load(config_file)
                setattr(self, category, SimpleNamespace(**config_dict))
#endregion

    #region: to_dict
    def to_dict(self):
        '''
        Convert the UnifiedConfiguration (self) into a nested dictionary.

        Returns
        -------
        dict
            A dictionary mapping each category name to the corresponding 
            settings.
        '''
        return {cat : self.category_to_dict(cat) for cat in self.__dict__}
    #endregion

    #region: category_to_dict
    def category_to_dict(self, category):
        '''
        Convert the configuration settings of the specified category to a 
        dictionary.

        Parameters
        ----------
        category : str
            The name of the category (e.g., 'model', 'path', 'plot') whose 
            configuration settings are to be converted to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the configuration settings for the 
            specified category.
        '''
        return getattr(self, category).__dict__
    #endregion

    # region: validate
    def validate(self):
        '''
        Validate the configuration settings.

        This method can be implemented to perform specific validation checks 
        on the configuration settings, ensuring they meet the expected 
        criteria.
        '''
        pass
    #endregion

    #region: set_defaults
    def set_defaults(self):
        '''
        Set default values for configuration settings.

        This method can be implemented to set default values for specific
        configuration settings that may not be provided in the configuration 
        files.
        '''
        pass
    #endregion

#region: parse_args
def parse_args():
    '''
    Parse command-line arguments for configuration loading

    The function defines and parses two command-line arguments:
    - `config_file`: A mandatory positional argument specifying the path to 
        the main configuration file.
    - `encoding`: An optional argument specifying the encoding of the 
        configuration file.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_file',
        type=str, 
        help='Path to the main configuration file',
        default=None
        )
    parser.add_argument(
        '-e', 
        '--encoding',
        type=str, 
        help='Encoding of the configuration files',
        default=None
    )
    return parser.parse_args()
#endregion

#region: config_from_cli_args
def config_from_cli_args():
    '''
    Load the configuration using command-line interface arguments.

    Returns
    -------
    UnifiedConfiguration
        An instance of UnifiedConfiguration initialized with the parsed 
        command-line arguments.
    '''
    args = parse_args()
    return UnifiedConfiguration(
        config_file=args.config_file, 
        encoding=args.encoding
        )
#endregion