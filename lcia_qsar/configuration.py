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

    Parameters
    ----------
    config_files_dict : dict
        Dictionary mapping categories to configuration file paths.

    Attributes
    ----------
    path : SimpleNamespace (optional)
        Configuration settings related to paths.
    model : SimpleNamespace (optional)
        Configuration settings related to models.
    plotting : SimpleNamespace (optional)
        Configuration settings related to plotting.

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

    VALID_CATEGORIES = {'path', 'model', 'plotting'}

    def __init__(self, config_files_dict):
        '''
        Initialize the UnifiedConfiguration object.

        Parameters
        ----------
        config_files_dict : dict
            Dictionary mapping categories to configuration file paths.
            Supported categories: 'path', 'model', 'plotting'.
        '''
        # Validate the input categories
        input_categories = set(config_files_dict)
        difference = input_categories - self.VALID_CATEGORIES
        if difference:
            raise ValueError(
                f'Invalid categories {difference}.\n'
                f'Expected: {self.VALID_CATEGORIES}')

        self._config_for_category = {}  # initialize

        # Load each file into its category
        for category, file_path in config_files_dict.items():
            with open(file_path, 'r') as config_file:
                config_dict = json.load(config_file)
                setattr(self, category, SimpleNamespace(**config_dict))
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

    # TODO: May become obsolete. Only used in the History class
    #region: estimator_names
    @property
    def estimator_names(self):
        return list(self.model.config_for_estimator)
    #endregion