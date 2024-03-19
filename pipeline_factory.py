'''
This module contains the `PipelineBuilder` class, which is used to build 
pipelines including estimators and preprocessors for different modeling 
workflows.

Example
-------
estimator_settings = {
    'RandomForestRegressor': {
        'module': 'sklearn.ensemble',
        'kwargs': {
            'max_features': 0.3333333333333333
        }
    }
}

preprocessor_settings = {
    'PowerTransformer': {
        'module': 'sklearn.preprocessing',
        'kwargs': {
            'standardize': False
        },
        'do_column_select': True
    }
}

builder = (
    PipelineBuilder(estimator_settings, preprocessor_settings, '_discrete')
    )
estimators = builder.instantiate_estimators()
'''

import importlib
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import numpy as np 

# TODO: If this is the only use of transform.py, migrate it here.
from transform import select_columns_without_pattern

#region: PipelineBuilder.__init__
class PipelineBuilder:
    '''
    A class to build pipelines for different modeling workflows.

    The class allows for the instantiation of estimator and preprocessing steps,
    and the creation of a complete modeling pipeline.

    Parameters
    ----------
    estimator_settings : dict
        Dictionary mapping estimator names to their corresponding settings.
    preprocessor_settings : dict
        Dictionary mapping preprocessor names to their corresponding settings.
    discrete_column_suffix : str
        Suffix pattern for identifying discrete feature columns.
    default_seed : int, optional (default=0)
        Default seed for random state generation for estimators.

    Attributes
    ----------
    estimator_settings : dict
        Configuration settings for the estimators.
    preprocessor_settings : dict
        Configuration settings for the preprocessors.
    _discrete_column_suffix : str
        Suffix pattern for discrete feature columns.
    _default_seed : int
        Default random seed.
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
        self.estimator_settings = estimator_settings
        self.preprocessor_settings = preprocessor_settings
        # TODO: Pattern handling for feature names could be more flexible
        self._discrete_column_suffix = discrete_column_suffix
        self._default_seed = default_seed
#endregion

    #region: instantiate_estimators
    def instantiate_estimators(self, preprocessor_names=None):
        '''
        Instantiate estimators with optional preprocessing steps.

        Parameters
        ----------
        preprocessor_names : list, optional
            List of names of preprocessors to include in the pipeline. If 
            None, all preprocessors in the settings will be included.

        Returns
        -------
        dict
            Dictionary mapping estimator names to instantiated estimator 
            objects.
        '''
        # Initialize the container.
        estimator_for_name = {}

        for name, config in self.estimator_settings.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})

            pre_steps = self.instantiate_preprocessors(preprocessor_names)                

            # Use the estimator-specific seed if provided, else use default
            seed = kwargs.pop('seed', self._default_seed)
            random_state = np.random.RandomState(seed=seed)
            final_estimator = getattr(module, class_name)(**kwargs)
            final_estimator.random_state = random_state
            
            estimator = make_pipeline(*pre_steps, final_estimator)
            estimator_for_name[name] = estimator

        return estimator_for_name
    #endregion

    #region: instantiate_preprocessors
    def instantiate_preprocessors(self, preprocessor_names=None):
        '''
        Instantiate preprocessing steps for the Pipeline.

        Parameters
        ----------
        preprocessor_names : list, optional
            List of names of the preprocessors to be instantiated. If None, 
            all preprocessors in the settings will be instantiated.

        Returns
        -------
        list
            List of instantiated preprocessor objects.
        '''   
        # Initialize the container.
        pre_steps = []
        for name, config in self.preprocessor_settings.items():

            # Skip preprocessors not in preprocessor_names if provided
            if preprocessor_names and name not in preprocessor_names:
                continue 

            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})
            preprocessor = getattr(module, class_name)(**kwargs)
            if config.get('do_column_select', False) is True:
                preprocessor = self._make_column_transformer(preprocessor)
            if hasattr(preprocessor, 'set_output'):
                # NOTE: See note in the docstring.
                preprocessor.set_output(transform='pandas')
            pre_steps.append(preprocessor)

        return pre_steps
    #endregion

    #region: _make_column_transformer
    def _make_column_transformer(self, transformer):
        '''
        Make a ColumnTransformer to "pass-through" discrete features.

        Parameters
        ----------
        transformer : object
            A transformer object to be used on continuous features.

        Returns
        -------
        ColumnTransformer
            A ColumnTransformer object that applies the given transformer to
            continuous features and passes through discrete features.
        '''
        select_continuous = select_columns_without_pattern(
            self._discrete_column_suffix+'$')  # matches at the end
        return make_column_transformer(
            (transformer, select_continuous),
            remainder='passthrough',
            verbose_feature_names_out=False,
            )    
    #endregion