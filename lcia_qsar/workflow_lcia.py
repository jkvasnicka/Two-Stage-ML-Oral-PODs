'''Provide support for managing workflows in the LCIA QSAR project.
'''

import os
import pandas as pd
import importlib
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from joblib import dump as joblib_dump

# TODO: May be a better way, e.g., sklearn imports 'from ..base import'
# Enable modules to be imported from the parent directory.
import sys
sys.path.append('..')
from common.workflow_base import SupervisedLearningWorkflow
from configuration import LciaQsarConfiguration
from history import LciaQsarHistory
from common.features import with_common_index 
from common.transform import select_columns_without_pattern
from common.evaluation import MetricWrapper

#region: LciaQsarModelingWorkflow.__init__
class LciaQsarModelingWorkflow(
        LciaQsarConfiguration, SupervisedLearningWorkflow, LciaQsarHistory):
    '''Take some data & instructions. Generate X & y. Learn an estimator's
    parameters and make predictions.
    '''
    def __init__(
            self, path_config_file, model_config_file):
        '''
        '''
        LciaQsarConfiguration.__init__(
            self, path_config_file, model_config_file)
        
        LciaQsarHistory.__init__(self)
        
        # TODO: Add the estimator level?
        self.instruction_names = [
            'target_effect', 
            'features_source', 
            'ld50_type', 
            'data_condition',
            'model_build'
        ]

        self.function_for_metric = self._instantiate_metrics()
#endregion
    
    #region: run
    def run(self, scores_to_csv=False, path_subdir=str()):
        '''Build and evaluate a model for each sequence of instructions.

        Save the results to disk.
        '''
        for keys in self.instruction_keys:
            key_for = dict(zip(self.instruction_names, keys))

            self.X, self.y = self.load_features_and_target(**key_for)            
            
            estimators = self._instantiate_estimators(key_for['data_condition'])
            for estimator in estimators:

                # Set the current estimator.
                self.estimator = estimator

                build_model = self._get_model_build_function(
                    key_for['model_build'])
                build_model()

        if scores_to_csv is True:
            self.write_score_histories_to_csv(path_subdir)

        # Write the workflow object to disk.
        self.dump()
    #endregion
    
    #region: _get_model_build_function
    def _get_model_build_function(self, model_build_key):
        '''
        '''
        function_name = '_build_model_' + model_build_key
        return getattr(self, function_name)
    #endregion
    
    #region: _build_model_with_selection
    def _build_model_with_selection(self):
        '''Augment the corresponding method of the base class.

        The results are saved to disk.
        '''
        base = SupervisedLearningWorkflow

        results_dict = base.build_model_with_selection(
            self, 
            self.feature_importance_scorings, 
            self.function_for_metric, 
            **self.kwargs_build_model
            )
        for k, v in results_dict.items():
            setattr(self, k, v)
    #endregion
             
    #region: _build_model_without_selection
    def _build_model_without_selection(self):
        '''Augment the corresponding method of the base class.

        The results are saved to disk.
        '''
        base = SupervisedLearningWorkflow

        self.performances = base.build_model_without_selection(
            self, 
            self.function_for_metric, 
            **self.kwargs_build_model
            )
    #endregion

    #region: load_features_and_target
    def load_features_and_target(
            self, *, target_effect, features_source, ld50_type, data_condition, 
            **kwargs):
        '''
        X, y are grouped together because they must share a common index.

        ** collects any unneeded key-value pairs from self.model_keys.
        '''
        X = self.load_features(
            features_source=features_source, 
            ld50_type=ld50_type, 
            data_condition=data_condition
            )

        y = self.load_target(target_effect=target_effect)

        # Use the intersection of chemicals.
        return with_common_index(X, y)
    #endregion

    #region: load_features
    def load_features(
            self, *, features_source, ld50_type, data_condition, **kwargs):
        '''Return the current features (X) as a pandas.DataFrame.
        '''
        features_path = self.file_for_features_source[features_source]
        X = pd.read_csv(features_path, index_col=0)

        if self.use_experimental_for_ld50[ld50_type] is True:
            ld50s_experimental = (
                pd.read_csv(
                    self.ld50_experimental_file, index_col=0).squeeze())
            X = LciaQsarModelingWorkflow._swap_column(
                X, 
                self.ld50_pred_column_for_source[features_source], 
                ld50s_experimental
                )
            
        if self.drop_missing_for_condition[data_condition] is True:
            # Use only samples with complete data.
            X = X.dropna(how='any')
        
        return X
    #endregion

    #region: load_target
    def load_target(self, *, target_effect, **kwargs):
        '''Return the current target variable (y) as a pandas.Series.
        '''
        ys = pd.read_csv(self.surrogate_pods_file, index_col=0)
        return ys[target_effect].squeeze().dropna()
    #endregion

    # TODO: Move to a PipelineBuilder class?
    #region: _instantiate_estimators
    def _instantiate_estimators(self, data_condition):
        '''For the current workflow instructions.
        '''
        # Initialize the container.
        estimators = []

        for name, config in self.config_for_estimator.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})

            pre_steps = self._instantiate_preprocessors(data_condition)                
            final_estimator = getattr(module, class_name)(**kwargs)
            # All final estimators receive the same seed.
            setattr(final_estimator, 'random_state', self.random_state_estimator)
            estimator = make_pipeline(*pre_steps, final_estimator)
            estimators.append(estimator)

        return estimators
    #endregion
    
    # TODO: Move to a PipelineBuilder class?
    #region: _instantiate_preprocessors
    def _instantiate_preprocessors(self, data_condition):
        '''Return a list of preprocessing steps for the Pipeline.

        Note
        -----
        There was a bug in sklearn.preprocessing.PowerTransformer which prevented
        configuring the transform() output globally via sklearn.set_config(). A 
        workaround is implemented below.
        '''    
        preprocessor_names = self.preprocessors_for_condition[data_condition]
    
        # Initialize the container.
        pre_steps = []
        for name in preprocessor_names:
            config = self.config_for_preprocessor[name]
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

    # TODO: Move to a PipelineBuilder class?
    #region: _make_column_transformer
    def _make_column_transformer(self, transformer):
        '''Make a ColumnTransformer to "pass-through" discrete features.
        '''
        select_continuous = select_columns_without_pattern(
            self.discrete_column_suffix+'$')  # matches at the end
        return make_column_transformer(
            (transformer, select_continuous),
            remainder='passthrough',
            verbose_feature_names_out=False,
            )    
    #endregion

    #region: _instantiate_metrics
    def _instantiate_metrics(self):
        '''
        Return a dictionary of scoring functions.
        '''        
        function_for_metric = {}

        for name, config in self.config_for_metric.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})
            
            metric = getattr(module, class_name)
            metric_instance = MetricWrapper(metric, **kwargs)

            function_for_metric[name] = metric_instance

        return function_for_metric
    #endregion

    # TODO: Define from configuration instead of hard-coding?
    #region: dump
    def dump(self, path_results_dir=str()):
        '''Save the current workflow to disk.
        '''
        workflow_file_name = 'workflow.joblib'
        workflow_file_path = os.path.join(
            path_results_dir, workflow_file_name)
        joblib_dump(self, workflow_file_path)
    #endregion

    #region: _swap_column
    @staticmethod
    def _swap_column(X, column_old, column_new):
        '''
        '''
        X = X.drop(column_old, axis=1)
        return pd.concat([X, column_new], axis=1)
    #endregion

    #region: get_important_features
    def get_important_features(self, model_key):
        '''Augment the method from the mix-in by providing arguments.
        '''
        kwargs = self.kwargs_build_model
        args = (
            kwargs['criterion_metric'],
            kwargs['n_features']
        )
        
        return LciaQsarHistory.get_important_features(
            self, 
            model_key,
            SupervisedLearningWorkflow.select_features, 
            args=args)
    #endregion

    #region: get_important_features_replicates
    def get_important_features_replicates(self, model_key):
        '''Augment the method from the mix-in by providing arguments.
        '''
        kwargs = self.kwargs_build_model
        stride = (
            kwargs['n_splits_select'] 
            * kwargs['n_repeats_select'] 
            * kwargs['n_repeats_perm']
        )
        args = (
            kwargs['criterion_metric'],
            kwargs['n_features']
        )

        return LciaQsarHistory.get_important_features_replicates(
            self,
            model_key,
            stride,
            SupervisedLearningWorkflow.select_features, 
            args=args)
    #endregion