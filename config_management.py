'''
This module provides a class to manage and access configuration settings 
related to different categories such as paths, models, and plotting. By 
unifying the configuration into a single object, it allows for a more 
streamlined access to settings throughout the code.

Classes
-------
- UnifiedConfiguration : A unified configuration object to manage various 
  settings.

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

#region: UnifiedConfiguration.__init__
class UnifiedConfiguration:
    '''
    This class provides a unified interface to access configuration settings 
    related to different categories such as paths, models, and plotting. The 
    configuration files for each category are loaded and made accessible as 
    attributes.

    Attributes
    ----------
    path : SimpleNamespace (optional)
        Configuration settings related to paths.
    model : SimpleNamespace (optional)
        Configuration settings related to models.
    plot : SimpleNamespace (optional)
        Configuration settings related to plotting.
    etc.

    Example
    -------
        config_files_dict = {
            'path': 'path-configuration.json',
            'model': 'model-configuration.json',
            'plot': 'plot-configuration.json'
        }
        config = UnifiedConfiguration(config_files_dict)
        model_settings = config.model
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

    def __init__(self, config_mapping_path, encoding='utf-8'):
        '''
        Initialize the UnifiedConfiguration object.

        Parameters
        ----------
        config_mapping_path : str
            Path to the JSON file mapping categories to configuration file 
            paths.
        encoding : str, optional
            Default is 'utf-8'.
        '''
        with open(config_mapping_path, 'r', encoding=encoding) as mapping_file:
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

    # TODO:
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

    # TODO: 
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