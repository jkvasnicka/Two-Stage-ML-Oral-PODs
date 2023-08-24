'''
This module provides the `WorkflowManager` class, which orchestrates the 
end-to-end process of data loading, preprocessing, model building, evaluation, 
and results management. It serves as the main driver of the modeling workflow, 
integrating various components like data management, pipeline building, 
feature selection, model evaluation, and results storage.

Example
-------
    config_mapping_path = 'configuration-mapping.json'
    config = UnifiedConfiguration(config_mapping_path)

    workflow_manager = WorkflowManager(config)
    workflow_manager.run()

Dependencies
------------
- `data_management` : Module for managing data.
- `pipeline_factory` : Module for building modeling pipelines.
- `feature_selection` : Module for feature selection.
- `model_factory` : Module for building models.
- `metrics_management` : Module for managing metrics.
- `model_evaluation` : Module for evaluating models.
- `model_key_creation` : Module for creating model keys.
- `results_management` : Module for managing results.
'''

from data_management import DataManager
from pipeline_factory import PipelineBuilder
from feature_selection import FeatureSelector
from model_factory import ModelBuilder
from metrics_management import MetricsManager
from model_evaluation import ModelEvaluator
from model_key_creation import ModelKeyCreator
from results_management import ResultsManager

#region: WorkflowManager.__init__
class WorkflowManager:
    '''
    This class serves as the main orchestrator of the modeling workflows. It 
    integrates various components like data management, pipeline building, 
    feature selection, model evaluation, and results storage to ensure the 
    entire process from data loading to results storage is seamless.

    Attributes
    ----------
    data_manager : DataManager
        Object to manage data loading and preprocessing.
    pipeline_builder : PipelineBuilder
        Object to build modeling pipelines.
    model_builder : ModelBuilder
        Object to train models.
    model_evaluator : ModelEvaluator
        Object to evaluate models.
    model_key_creator : ModelKeyCreator
        Object to create model keys.
    results_manager : ResultsManager
        Object to manage results storage and retrieval.
    '''
    def __init__(self, config):
        '''
        Initialize the `WorkflowManager`.

        Parameters
        ----------
        config : UnifiedConfiguration
            Container for all configuration settings for the workflows.
        '''
        self._config = config 

        self.data_manager = DataManager(
            config.data,
            config.path
            )

        self.pipeline_builder = PipelineBuilder(
            config.category_to_dict('estimator'), 
            config.preprocessor.settings,
            config.data.discrete_column_suffix
            )

        feature_selector = FeatureSelector(
            config.feature_selection, 
            config.model.n_jobs
            )

        self.model_builder = ModelBuilder(feature_selector)

        metrics_manager = MetricsManager(config.category_to_dict('metric'))
        self.model_evaluator = ModelEvaluator(
            config.evaluation, 
            metrics_manager,
            feature_selector=feature_selector,
            n_jobs=config.model.n_jobs
            )

        self.model_key_creator = ModelKeyCreator(
            config.model.modeling_instructions
            )
        self.results_manager = ResultsManager(
            config.path.results_dir,
            self.model_key_creator
        )
#endregion
    
    #region: run
    def run(self):
        '''
        Execute the modeling workflows based on the provided configuration. 

        This method orchestrates the entire process of loading data, building 
        and evaluating models, and storing the results. It reads the modeling 
        instructions from the configuration, processes each instruction, and 
        leverages other components to execute each step of the workflows.

        Returns
        -------
        None
        '''
        # For reproducibility,
        self.results_manager.write_configuration(self._config)

        for instruction in self._config.model.modeling_instructions:

            X, y = self.data_manager.load_features_and_target(**instruction)            
            
            preprocessor_names = (
                self._config.preprocessor.preprocessors_for_condition[
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
                
                model_key = self.model_key_creator.create_model_key(
                    instruction, 
                    estimator_name
                    )                
                self.results_manager.write_results(model_key, all_results)
    #endregion

    #region: _process_instruction
    def _process_instruction(self, instruction, X, y, estimator):
        '''
        Process a single modeling instruction by evaluating and building a 
        model.

        Given an instruction, this method first evaluates the model using 
        cross-validation and then builds the final model on the full dataset. 
        It's crucial that the model is evaluated using cross-validation BEFORE 
        it's built on the full dataset. Evaluation results and model 
        parameters are then returned.

        Parameters
        ----------
        instruction : dict
            Dictionary containing the modeling instruction.
        X : pandas.DataFrame
            Features for the model.
        y : pandas.Series
            Target variable for the model.
        estimator : object
            The model estimator or pipeline to be trained and evaluated.

        Returns
        -------
        dict
            Dictionary containing the evaluation results and model parameters.
        '''
        # Determine whether to perform feature selection
        select_features = instruction['select_features'] == 'true'

        evaluation_results = self.model_evaluator.cross_validate_model(
                estimator, 
                X, 
                y,
                select_features
        )
        
        build_results = self.model_builder.train_final_model(
            estimator, 
            X, 
            y, 
            select_features
        )
        
        return {**evaluation_results, **build_results}
    #endregion