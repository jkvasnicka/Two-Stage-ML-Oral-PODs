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

        self.data_manager = DataManager(
            self.config.data,
            self.config.path
            )

        self.pipeline_builder = PipelineBuilder(
            self.config.to_dict('estimator'), 
            self.config.preprocessor.settings,
            self.config.data.discrete_column_suffix
            )

        feature_selector = FeatureSelector(
            self.config.feature_selection, 
            self.config.model.n_jobs
            )

        self.model_builder = ModelBuilder(feature_selector)

        metrics_manager = MetricsManager(self.config.to_dict('metric'))
        self.model_evaluator = ModelEvaluator(
            self.config.evaluation, 
            metrics_manager,
            feature_selector=feature_selector,
            n_jobs=self.config.model.n_jobs
            )
        
        self.results_manager = ResultsManager(
            output_dir=self.config.path.results_dir)
#endregion
    
    #region: run
    def run(self):
        '''
        '''
        # TODO: ModelKeyManager?
        model_key_names = [
            k for k in self.config.model.modeling_instructions[0].keys() 
            if k != 'estimators'
            ]
        model_key_names.append('estimator')
        self.results_manager.write_model_key_names(model_key_names)

        for instruction in self.config.model.modeling_instructions:

            X, y = self.data_manager.load_features_and_target(**instruction)            
            
            preprocessor_names = (
                self.config.preprocessor.preprocessors_for_condition[
                    instruction['data_condition']]
            )
            estimator_for_name = (
                self.pipeline_builder.instantiate_estimators(
                preprocessor_names)
            )
            
            for estimator_name in instruction['estimators']:
                estimator = estimator_for_name[estimator_name]

                all_results = self._process_instruction(
                    instruction, 
                    X, 
                    y, 
                    estimator
                    )
                
                model_key = self._construct_model_key(instruction, estimator_name)                
                self.results_manager.write_results(model_key, all_results)
    #endregion

    #region: _process_instruction
    def _process_instruction(self, instruction, X, y, estimator):
        '''Evaluate and then build a model for each set of instructions. 

        It's crucial that the model is evaluated using cross-validation BEFORE 
        it's built on the full dataset.
        '''
        # Determine whether to perform feature selection
        with_selection = instruction['model_build'] == 'with_selection'

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
        
        return {**evaluation_results, **build_results}
    #endregion

    #region: _construct_model_key
    def _construct_model_key(self, instruction, estimator_name):
        '''
        Define a unique identifier for the model
        '''
        model_key = [v for k, v in instruction.items() if k != 'estimators']
        model_key.append(estimator_name)
        return model_key
    #endregion