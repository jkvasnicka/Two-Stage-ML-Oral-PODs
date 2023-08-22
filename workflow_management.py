'''
'''

from data_management import DataManager
from pipeline_factory import PipelineBuilder
from feature_selection import FeatureSelector
from model_factory import ModelBuilder
from metrics_management import MetricsManager
from model_evaluation import ModelEvaluator
from results_management import ResultsManager

#region: WorkflowManager.__init__
class WorkflowManager:
    '''
    '''
    def __init__(self, config):
        '''
        '''
        self.config = config 

        self.data_manager = DataManager(self.config)

        self.pipeline_builder = PipelineBuilder(
            self.config.to_dict('estimator'), 
            self.config.to_dict('preprocessor'),
            self.config.model.discrete_column_suffix
            )

        feature_selector = FeatureSelector(self.config.model)

        self.model_builder = ModelBuilder(feature_selector)

        metrics_manager = MetricsManager(self.config.to_dict('metric'))
        self.model_evaluator = ModelEvaluator(
            self.config.model, 
            metrics_manager,
            feature_selector
            )
        
        self.results_manager = ResultsManager(
            output_dir=self.config.path.results_dir)

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
        '''Evaluate and then build a model for each set of instructions. 

        It's crucial that the model is evaluated using cross-validation BEFORE 
        it's built on the full dataset.
        '''
        self.results_manager.write_model_key_names(
            self.config.model.model_key_names)

        for instruction_key in self.config.model.instruction_keys:
            key_for = dict(zip(self.instruction_names, instruction_key))

            X, y = self.data_manager.load_features_and_target(**key_for)            
            
            preprocessor_names = (
                self.config.model.preprocessors_for_condition[
                    key_for['data_condition']]
            )
            estimator_for_name = (
                self.pipeline_builder.instantiate_estimators(
                preprocessor_names)
            )
            
            for est_name, estimator in estimator_for_name.items():
                # Define a unique identifier for the model
                model_key = (*instruction_key, est_name)

                if key_for['model_build'] == 'with_selection':
                    with_selection = True 
                else: 
                    with_selection = False
                
                evaluation_results = self.model_evaluator.cross_validate_model(
                        estimator, 
                        X, 
                        y,
                        with_selection
                        )
                
                build_results = self.model_builder.train_final_model(
                    estimator, 
                    X, 
                    y, 
                    with_selection
                    )
                
                all_results = {**evaluation_results, **build_results}

                self.results_manager.write_results(model_key, all_results)
    #endregion