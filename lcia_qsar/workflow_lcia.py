'''Provide support for managing workflows in the LCIA QSAR project.
'''

import os
import numpy as np
import importlib
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from joblib import dump as joblib_dump

# TODO: May be a better way, e.g., sklearn imports 'from ..base import'
# Enable modules to be imported from the parent directory.
import sys
sys.path.append('..')
from common.workflow_base import SupervisedLearningWorkflow
from history import LciaQsarHistory
from common.transform import select_columns_without_pattern
from common.evaluation import MetricWrapper

from data_management import DataManager
from results_management import ResultsManager

#region: LciaQsarModelingWorkflow.__init__
class LciaQsarModelingWorkflow(
        SupervisedLearningWorkflow, LciaQsarHistory):
    '''Take some data & instructions. Generate X & y. Learn an estimator's
    parameters and make predictions.
    '''
    def __init__(self, config):
        '''
        '''
        self.config = config 

        # TODO: Move to separate `EstimatorInstantiator` class
        self.random_state_estimator = np.random.RandomState(seed=self.config.model.seed_estimator)
        
        LciaQsarHistory.__init__(self)
        
        # TODO: Where should this go?
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
        data_manager = DataManager(self.config)
        results_manager = ResultsManager(output_dir=self.config.path.results_dir)
        results_manager.write_model_key_names(self.config.model.model_key_names)

        for instruction_key in self.config.model.instruction_keys:
            key_for = dict(zip(self.instruction_names, instruction_key))

            self.X, self.y = data_manager.load_features_and_target(**key_for)            
            
            estimator_for_name = self._instantiate_estimators(key_for['data_condition'])
            for est_name, estimator in estimator_for_name.items():
                # Define a unique identifier for the model
                model_key = (*instruction_key, est_name)

                # Set the current estimator.
                self.estimator = estimator

                build_model = self._get_model_build_function(
                    key_for['model_build'])
                build_model(results_manager, model_key)

                results_manager.write_estimator(estimator, model_key)

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
    def _build_model_with_selection(self, results_manager, model_key):
        '''Augment the corresponding method of the base class.

        The results are saved to disk.
        '''
        base = SupervisedLearningWorkflow

        results_dict = base.build_model_with_selection(
            self, 
            self.config.model.feature_importance_scorings, 
            self.function_for_metric, 
            **self.config.model.kwargs_build_model
            )
        for result_type, df in results_dict.items():
            setattr(self, result_type, df)
            results_manager.write_result(df, model_key, result_type)
    #endregion
             
    #region: _build_model_without_selection
    def _build_model_without_selection(self, results_manager, model_key):
        '''Augment the corresponding method of the base class.

        The results are saved to disk.
        '''
        base = SupervisedLearningWorkflow

        self.performances = base.build_model_without_selection(
            self, 
            self.function_for_metric, 
            **self.config.model.kwargs_build_model
            )
        results_manager.write_result(self.performances, model_key, 'performances')
    #endregion

    # TODO: Move to a PipelineBuilder class?
    #region: _instantiate_estimators
    def _instantiate_estimators(self, data_condition):
        '''For the current workflow instructions.
        '''
        # Initialize the container.
        estimator_for_name = {}

        for name, config in self.config.model.config_for_estimator.items():
            module = importlib.import_module(config['module'])
            class_name = config.get('class', name)
            kwargs = config.get('kwargs', {})

            pre_steps = self._instantiate_preprocessors(data_condition)                
            final_estimator = getattr(module, class_name)(**kwargs)
            # All final estimators receive the same seed.
            setattr(final_estimator, 'random_state', self.random_state_estimator)
            estimator = make_pipeline(*pre_steps, final_estimator)
            estimator_for_name[name] = estimator

        return estimator_for_name
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
        preprocessor_names = self.config.model.preprocessors_for_condition[data_condition]
    
        # Initialize the container.
        pre_steps = []
        for name in preprocessor_names:
            config = self.config.model.config_for_preprocessor[name]
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
            self.config.model.discrete_column_suffix+'$')  # matches at the end
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

        for name, config in self.config.model.config_for_metric.items():
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

    # TODO: Move to ResultsAnalyzer class.
    #region: get_important_features
    def get_important_features(self, model_key):
        '''Augment the method from the mix-in by providing arguments.
        '''
        kwargs = self.config.model.kwargs_build_model
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

    # TODO: Move to ResultsAnalyzer class.
    #region: get_important_features_replicates
    def get_important_features_replicates(self, model_key):
        '''Augment the method from the mix-in by providing arguments.
        '''
        kwargs = self.config.model.kwargs_build_model
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