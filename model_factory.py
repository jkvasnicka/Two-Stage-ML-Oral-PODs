'''
This module contains the ModelBuilder class, which is responsible for building 
machine learning models with or without feature selection. These models can be 
used for out-of-sample prediction and generalization error evaluation through 
cross-validation.

Example
-------
feature_selector = FeatureSelector(model_settings)
model_builder = ModelBuilder(feature_selector)
result = model_builder.build(estimator, X, y, select_features=True)
'''

#region: ModelBuilder.__init__
class ModelBuilder:
    '''
    A class to handle the construction of a machine learning model, with 
    optional feature selection.

    Attributes
    ----------
    feature_selector : FeatureSelector, optional
        An instance of the FeatureSelector class to perform feature selection 
        during model building, default is None.
    '''
    def __init__(self, feature_selector=None):
        '''
        Initialize the ModelBuilder.

        Parameters
        ----------
        feature_selector : FeatureSelector, optional
            The feature selector to use if feature selection is required during 
            model building.
        '''
        self.feature_selector = feature_selector
#endregion
    
    #region: train_final_model
    def train_final_model(self, estimator, X, y, select_features=False):
        '''
        Build a model with or without feature selection.

        Parameters
        ----------
        estimator : object
            A machine learning estimator that follows the scikit-learn API.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        select_features : bool, optional
            Whether to perform feature selection during model building, 
            default is False.

        Returns
        -------
        dict
            A dictionary containing the built estimator and additional 
            results.
        '''
        if select_features:
            return self._train_final_model_with_selection(
                estimator, X, y)
        else:
            return self._train_final_model_without_selection(
                estimator, X, y)
    #endregion

    #region: _train_final_model_with_selection
    def _train_final_model_with_selection(self, estimator, X, y):
        '''
        Private method to build a model with feature selection.

        Parameters
        ----------
        estimator : object
            A machine learning estimator that follows the scikit-learn API.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.

        Returns
        -------
        dict
            A dictionary containing the built estimator, important features, 
            and importances.
        '''
        estimator, important_features, importances = (
            self.feature_selector.nested_feature_selection(estimator, X, y)
        )
        estimator.fit(X[important_features], y)

        # TODO: Create a Results class?
        build_results = {
            'estimator' : estimator, 
            'important_features' : important_features,  # TODO: Write?
            'importances' : importances
        }
        return build_results
    #endregion
    
    #region: _train_final_model_without_selection
    def _train_final_model_without_selection(self, estimator, X, y):
        '''
        Private method to build a model without feature selection.

        Parameters
        ----------
        estimator : object
            A machine learning estimator that follows the scikit-learn API.
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.

        Returns
        -------
        dict
            A dictionary containing the built estimator.
        '''
        estimator.fit(X, y)

        # TODO: Create a Results class?
        build_results = {
            'estimator' : estimator
        }
        return build_results
    #endregion