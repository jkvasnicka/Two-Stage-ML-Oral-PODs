'''Provide support for managing workflows in the LCIA QSAR project.
'''

import numpy as np
from data_management import DataManager
from metrics_management import MetricsManager
from pipeline_factory import PipelineBuilder
from workflow_base import SupervisedLearningWorkflow
from results_management import ResultsManager

#region: LciaQsarModelingWorkflow.__init__
class LciaQsarModelingWorkflow(SupervisedLearningWorkflow):
    '''Take some data & instructions. Generate X & y. Learn an estimator's
    parameters and make predictions.
    '''
    def __init__(self, config):
        '''
        '''
        self.config = config 

        # TODO: model_evaluator = ModelEvaluator(metrics_manager)
        self.metrics_manager = MetricsManager(self.config.to_dict('metrics'))

        self.pipeline_builder = PipelineBuilder(
            self.config.to_dict('estimator'), 
            self.config.to_dict('preprocessor'),
            self.config.model.discrete_column_suffix
            )
                
        # TODO: Move to separate `EstimatorInstantiator` class
        self.random_state = np.random.RandomState(seed=0)

        # TODO: Where should this go?
        self.instruction_names = [
            'target_effect', 
            'features_source', 
            'ld50_type', 
            'data_condition',
            'model_build'
        ]
#endregion
    
    #region: run
    def run(self):
        '''Build and evaluate a model for each sequence of instructions.

        Save the results to disk.
        '''
        # TODO: Move these to __init__?
        data_manager = DataManager(self.config)
        results_manager = ResultsManager(output_dir=self.config.path.results_dir)
        results_manager.write_model_key_names(self.config.model.model_key_names)

        for instruction_key in self.config.model.instruction_keys:
            key_for = dict(zip(self.instruction_names, instruction_key))

            self.X, self.y = data_manager.load_features_and_target(**key_for)            
            
            # TODO: Allow optional subset of preprocessors for data condition
            estimator_for_name = self.pipeline_builder.instantiate_estimators(self.random_state)
            
            for est_name, estimator in estimator_for_name.items():
                # Define a unique identifier for the model
                model_key = (*instruction_key, est_name)

                # Set the current estimator.
                self.estimator = estimator

                build_model = self._get_model_build_function(
                    key_for['model_build'])
                build_model(results_manager, model_key)

                results_manager.write_estimator(estimator, model_key)
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
            **self.config.model.kwargs_build_model
            )
        results_manager.write_result(self.performances, model_key, 'performances')
    #endregion