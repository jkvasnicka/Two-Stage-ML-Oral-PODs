'''Define global configuration settings for the LCIA QSAR Project.
'''

from json import loads
from os import path
import numpy as np


#region: BaseConfiguration.__init__
class BaseConfiguration:
    '''
    '''
    def __init__(self, *config_files):
        for config_file in config_files:
            self._set_configuration_attributes(config_file)
#endregion

    #region: _set_configuration_attributes
    def _set_configuration_attributes(self, config_file):
        '''
        '''
        config_dict = BaseConfiguration.load_configuration_file(config_file)
        for k, v in config_dict.items():
            setattr(self, k, v)
    #endregion

    #region: load_configuration_file
    @staticmethod
    def load_configuration_file(config_file, prefix=str(), prefix_contents=False):
        '''Load the configuration JSON file as a dictionary.
        '''
        config_file = path.join(prefix, config_file)

        with open(config_file) as f:
            config_dict = loads(f.read())

        if prefix_contents is True:
            config_dict = BaseConfiguration.transform_recursively(
                config_dict, lambda v : path.join(prefix, v))
        
        return config_dict
    #endregion

    #region: transform_recursively
    @staticmethod
    def transform_recursively(d, func):
        '''Traverse a possibly-nested dictionary, depth first, and apply a 
        function to all its terminal nodes/values.

        Returns
        -------
        dict    
        '''
        # The "base case" is covered by the "else" condition in the following.
        if isinstance(d, dict):
            # The following will eventually be the final return.
            return {k: BaseConfiguration.transform_recursively(v, func) 
                    for k, v in d.items()}
        if isinstance(d, list):
            return [BaseConfiguration.transform_recursively(v, func) 
                    for v in d]
        else:
            return func(d)
    #endregion

#region: LciaQsarConfiguration.__init__
class LciaQsarConfiguration(BaseConfiguration):
    '''Mix-in class, provides configuration attributes and related logic.
    '''
    def __init__(self, path_config_file, model_config_file):
        super().__init__(path_config_file, model_config_file)
#endregion

    #region: instruction_keys
    @property
    def instruction_keys(self):
        return self._instruction_keys
    @instruction_keys.setter
    def instruction_keys(self, instructions):
        '''
        '''
        if not self._is_nested_sequence(instructions):
            instructions = [instructions]
            
        self._instruction_keys = [tuple(keys) for keys in instructions]
    #endregion

    #region: _is_nested_sequence
    @staticmethod
    def _is_nested_sequence(instructions):
        '''
        '''
        return isinstance(instructions, (list, tuple)) and \
            all(isinstance(value, (list, tuple)) for value in instructions)
    #endregion

    #region: estimator_names
    @property
    def estimator_names(self):
        return list(self.config_for_estimator)
    #endregion

    #region: random_state_estimator & seed_estimator
    @property
    def random_state_estimator(self):
        return self._random_state_estimator

    @property
    def seed_estimator(self):
        return self._random_state_estimator.get_state()[1][0]

    @seed_estimator.setter
    def seed_estimator(self, seed):
        self._random_state_estimator = np.random.RandomState(seed=seed)
    #endregion