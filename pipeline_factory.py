'''
'''

import importlib
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import numpy as np 

from transform import select_columns_without_pattern

class PipelineBuilder:
    '''
    '''
    def __init__(
            self, 
            estimator_settings, 
            preprocessor_settings, 
            discrete_column_suffix, 
            default_seed=0
            ):
        '''
        '''
        self.estimator_settings = estimator_settings  # dict
        self.preprocessor_settings = preprocessor_settings  # dict
        # TODO: Pattern handling for feature names could be more flexible
        self._discrete_column_suffix = discrete_column_suffix
        self._default_seed = default_seed

    #region: instantiate_estimators
    def instantiate_estimators(self):
        '''For the current workflow instructions.
        '''
        # Initialize the container.
        estimator_for_name = {}

        for name, config in self.estimator_settings.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})

            pre_steps = self.instantiate_preprocessors()                

            # Use the estimator-specific seed if provided, else use the default
            seed = kwargs.pop('seed', self._default_seed)
            random_state = np.random.RandomState(seed=seed)
            final_estimator = getattr(module, class_name)(**kwargs)
            final_estimator.random_state = random_state
            
            estimator = make_pipeline(*pre_steps, final_estimator)
            estimator_for_name[name] = estimator

        return estimator_for_name
    #endregion

    #region: instantiate_preprocessors
    def instantiate_preprocessors(self):
        '''Return a list of preprocessing steps for the Pipeline.

        Note
        -----
        There was a bug in sklearn.preprocessing.PowerTransformer which prevented
        configuring the transform() output globally via sklearn.set_config(). A 
        workaround is implemented below.
        '''        
        # Initialize the container.
        pre_steps = []
        for name, config in self.preprocessor_settings.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})
            preprocessor = getattr(module, class_name)(**kwargs)
            if config.get('do_column_select', False) is True:
                preprocessor = self.make_column_transformer(preprocessor)
            if hasattr(preprocessor, 'set_output'):
                # NOTE: See note in the docstring.
                preprocessor.set_output(transform='pandas')
            pre_steps.append(preprocessor)

        return pre_steps
    #endregion

    #region: make_column_transformer
    def make_column_transformer(self, transformer):
        '''Make a ColumnTransformer to "pass-through" discrete features.
        '''
        select_continuous = select_columns_without_pattern(
            self._discrete_column_suffix+'$')  # matches at the end
        return make_column_transformer(
            (transformer, select_continuous),
            remainder='passthrough',
            verbose_feature_names_out=False,
            )    
    #endregion